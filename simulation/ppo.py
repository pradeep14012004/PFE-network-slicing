# Agent Deep Reinforcement Learning centralisé pour l'allocation dynamique de bande passante
# dans un réseau SDN avec double contrôleur Ryu et simulation Mininet
# Architecture: Algo1 (PPO ON/OFF) → Algo2 (SDN Ctrl1: seuils) → Algo3 (SDN Ctrl2: routage/débordement) → Algo4 (sécurité/réallocation)

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import logging
import os
import csv
import requests
import time
from collections import deque
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from torch.distributions import Dirichlet, Bernoulli
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration & SLA
# ---------------------------------------------------------------------------

@dataclass
class NetworkConfig:
    """Configuration du réseau SDN"""
    ryu_controller_ip: str = field(default_factory=lambda: os.getenv("RYU_CONTROLLER_IP", "172.18.0.10"))
    ryu_rest_port: int = field(default_factory=lambda: int(os.getenv("RYU_REST_PORT", "8080")))
    num_switches: int = 1
    num_ports_per_switch: int = 3
    # 4 slices: 0=URLLC, 1=URLLC_eMBB_MIX, 2=eMBB, 3=mMTC
    num_slices: int = 4
    max_bandwidth_mbps: int = 750
    observation_interval: float = 2.0
    alpha_seuil: float = 0.1     # Seuil minimal (fraction de Max_i) avant extinction

    # EcoSlice1 = URLLC (idx 0) toujours ON ; EcoSlice2 = URLLC_eMBB_MIX (idx 1) secours
    eco_slice_ids: Tuple[int, ...] = (0, 1)
    eco1_capacity_mbps: float = 300.0
    eco2_capacity_mbps: float = 300.0
    eco1_overflow_threshold: float = 0.75    # 75% → activer EcoSlice2

    # Noms et priorité de routage (ordre décroissant de criticité)
    # 0=URLLC > 1=URLLC_eMBB_MIX > 3=mMTC > 2=eMBB
    slice_names: Tuple[str, ...] = ("URLLC", "URLLC_eMBB_MIX", "eMBB", "mMTC")
    slice_priority: Tuple[int, ...] = (0, 1, 3, 2)   # indices in priority order

    # tc class ids matching topology.py
    tc_classes: Tuple[str, ...] = ("1:10", "1:11", "1:12", "1:13")

    # Optional proactive demand forecast. If forecast_csv is set, PPO can use
    # q10/q50/q90 Mbps forecasts exported from the forecasting benchmark.
    forecast_csv: str = field(default_factory=lambda: os.getenv("FORECAST_CSV", ""))
    forecast_quantile: str = field(default_factory=lambda: os.getenv("FORECAST_QUANTILE", "q90"))
    forecast_mode: str = field(default_factory=lambda: os.getenv("FORECAST_MODE", "max"))
    forecast_loop: bool = field(default_factory=lambda: os.getenv("FORECAST_LOOP", "1") not in {"0", "false", "False"})


@dataclass
class SLARequirements:
    """Définition des SLA pour chaque slice"""
    slice_id: int
    min_bandwidth_mbps: float
    max_latency_ms: float
    max_packet_loss: float


# ---------------------------------------------------------------------------
# Résultats intermédiaires (structures de données partagées entre étapes)
# ---------------------------------------------------------------------------

@dataclass
class TrafficRedirectResult:
    """Résultat produit par RéacheminementTraffic (Algos 2, 3, 4)"""
    c_filtre: np.ndarray          = field(default_factory=lambda: np.array([]))
    c_final: np.ndarray           = field(default_factory=lambda: np.array([]))
    rho: np.ndarray               = field(default_factory=lambda: np.array([]))
    delta_E: float                = 0.0
    L: float                      = 0.0
    eta_b: float                  = 0.0
    surcharge: bool               = False
    eco2_active: bool             = False
    v_rediriger: List[Tuple[int, float]] = field(default_factory=list)
    load_eco1: float              = 0.0
    load_eco2: float              = 0.0


class ForecastDemandProvider:
    """Sequential slice-level demand forecasts loaded from a CSV file."""

    def __init__(
        self,
        csv_path: str,
        slice_names: Tuple[str, ...],
        quantile: str = "q90",
        loop: bool = True,
    ):
        self.csv_path = csv_path
        self.slice_names = slice_names
        self.quantile = quantile
        self.loop = loop
        self.column = f"{quantile}_mbps"
        self.cursor = 0
        self.timestamps: List[str] = []
        self.loads: List[np.ndarray] = []
        self.current_timestamp: Optional[str] = None
        self._load_csv()

    def _load_csv(self) -> None:
        by_timestamp: Dict[str, np.ndarray] = {}
        with open(self.csv_path, newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if self.column not in (reader.fieldnames or []):
                raise ValueError(
                    f"Forecast CSV {self.csv_path} is missing column {self.column}. "
                    "Expected columns such as q10_mbps, q50_mbps, q90_mbps."
                )
            for row in reader:
                timestamp = row["timestamp"]
                slice_name = row["slice"]
                if slice_name not in self.slice_names:
                    continue
                if timestamp not in by_timestamp:
                    by_timestamp[timestamp] = np.zeros(len(self.slice_names), dtype=np.float32)
                idx = self.slice_names.index(slice_name)
                by_timestamp[timestamp][idx] = max(0.0, float(row[self.column]))

        self.timestamps = sorted(by_timestamp)
        self.loads = [by_timestamp[timestamp] for timestamp in self.timestamps]
        if not self.loads:
            raise ValueError(f"No usable forecast rows found in {self.csv_path}")

    def next_loads(self) -> np.ndarray:
        if self.cursor >= len(self.loads):
            if not self.loop:
                self.cursor = len(self.loads) - 1
            else:
                self.cursor = 0
        self.current_timestamp = self.timestamps[self.cursor]
        loads = self.loads[self.cursor].copy()
        self.cursor += 1
        return loads

    def combine(self, observed_loads: np.ndarray, mode: str) -> Tuple[np.ndarray, np.ndarray]:
        forecast_loads = self.next_loads()
        if mode == "forecast":
            return forecast_loads, forecast_loads
        if mode == "add":
            return observed_loads + forecast_loads, forecast_loads
        if mode == "max":
            return np.maximum(observed_loads, forecast_loads), forecast_loads
        raise ValueError("FORECAST_MODE must be one of: max, forecast, add")


# ---------------------------------------------------------------------------
# Environnement SDN
# ---------------------------------------------------------------------------

class SDNEnvironment:
    def __init__(self, config: NetworkConfig):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.config = config

        self.sla_requirements = {
            0: SLARequirements(slice_id=0, min_bandwidth_mbps=10,  max_latency_ms=1,   max_packet_loss=0.01),  # URLLC
            1: SLARequirements(slice_id=1, min_bandwidth_mbps=50,  max_latency_ms=2,   max_packet_loss=0.05),  # URLLC_eMBB_MIX
            2: SLARequirements(slice_id=2, min_bandwidth_mbps=100, max_latency_ms=10,  max_packet_loss=0.1),   # eMBB
            3: SLARequirements(slice_id=3, min_bandwidth_mbps=20,  max_latency_ms=50,  max_packet_loss=0.2),   # mMTC
        }

        self.base_url = f"http://{config.ryu_controller_ip}:{config.ryu_rest_port}"
        self.ports = self.wait_for_ports()
        self.numports = len(self.ports)

        # Historique des pics de trafic par slice (pour le seuil αseuil·Maxi)
        self.traffic_peak: Dict[int, float] = {i: 1.0 for i in range(config.num_slices)}

        # Énergie de référence (station entièrement allumée) — on l'estime au premier pas
        self.f_base: Optional[float] = None

        logger.info(f"Environnement SDN initialisé - Contrôleur: {self.base_url}")

    # ------------------------------------------------------------------
    # Appels REST
    # ------------------------------------------------------------------

    def get_all_ports(self) -> List[List[str]]:
        url = f"{self.base_url}/getports"
        try:
            response = requests.get(url, timeout=5)
            response.raise_for_status()
            return response.json().get('ports', [])
        except requests.exceptions.RequestException as e:
            logger.error(f"Erreur requête REST {url}: {e}")
            return []

    def wait_for_ports(self, retries: int = 60, delay: float = 2.0) -> List[str]:
        """Wait until Mininet has registered switch interfaces in Ryu."""
        for attempt in range(1, retries + 1):
            ports_payload = self.get_all_ports()
            if ports_payload and ports_payload[0]:
                ports = ports_payload[0]
                self.logger.info(f"Ports détectés depuis Ryu: {ports}")
                return ports
            self.logger.info(
                f"Ports Ryu indisponibles, attente Mininet "
                f"({attempt}/{retries})..."
            )
            time.sleep(delay)
        raise RuntimeError(
            "Aucun port reçu depuis /getports. Lance d'abord topology.py dans "
            "le conteneur Mininet et vérifie que Ryu est joignable."
        )

    def _make_rest_request(self, endpoint: str, method: str = "GET", data: dict = None) -> Optional[dict]:
        url = f"{self.base_url}{endpoint}"
        try:
            r = requests.get(url, timeout=5) if method == "GET" else requests.post(url, json=data, timeout=5)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            self.logger.error(f"REST error {method} {url}: {e}")
            return None

    # ------------------------------------------------------------------
    # État réseau
    # ------------------------------------------------------------------

    def get_state(self) -> np.ndarray:
        resp = self._make_rest_request("/getstats")
        if not resp or 'stats' not in resp:
            return np.zeros(self.config.num_ports_per_switch * self.config.num_slices * 4)
        stats = resp['stats']
        state = []
        for port in self.ports:
            port_stats = [s for s in stats if s['interface'] == port]
            for cls_idx, cls in enumerate(self.config.tc_classes):
                stat = next((x for x in port_stats if x['class'] == cls), None)
                if stat:
                    sla = self.sla_requirements[cls_idx]
                    state.extend([
                        (stat['dropped'] / max(1, stat['nbre_demands'])),
                        min(1, stat['rate'] / self.config.max_bandwidth_mbps),
                        max(0, 1 - stat['latency'] / sla.max_latency_ms),
                        stat['throughput'] / max(1e-6, stat['rate']),
                    ])
                    self.traffic_peak[cls_idx] = max(self.traffic_peak[cls_idx], stat['rate'])
                else:
                    state.extend([0.0, 0.0, 0.0, 0.0])
        return np.array(state, dtype=np.float32)

    def get_traffic_loads(self) -> np.ndarray:
        """Retourne la charge de trafic actuelle par slice (en Mbps)."""
        resp = self._make_rest_request("/getstats")
        loads = np.zeros(self.config.num_slices)
        if not resp or 'stats' not in resp:
            return loads
        stats = resp['stats']
        for port in self.ports:
            port_stats = [s for s in stats if s['interface'] == port]
            for cls_idx, cls in enumerate(self.config.tc_classes):
                stat = next((x for x in port_stats if x['class'] == cls), None)
                if stat:
                    loads[cls_idx] += stat['rate']
        return loads

    def get_energy_consumption(self) -> float:
        """Consommation énergétique courante de la station (valeur normalisée)."""
        resp = self._make_rest_request("/getenergy")
        if resp and 'power_w' in resp:
            return float(resp['power_w'])
        return 0.0

    # ------------------------------------------------------------------
    # Application des actions
    # ------------------------------------------------------------------

    def apply_actions(self, actions: np.ndarray) -> None:
        """Applique un vecteur de taux de bande passante (actions normalisées [0,1])."""
        for i, port in enumerate(self.ports):
            rates = actions[i * self.config.num_slices:(i + 1) * self.config.num_slices] \
                    * (self.config.max_bandwidth_mbps * self.config.num_ports_per_switch)
            data = {"id": port, "rates": rates.tolist()}
            self._make_rest_request("/setaction", method="POST", data=data)

    def apply_activation(self, c_final: np.ndarray, rho: np.ndarray) -> None:
        """Applique l'activation ON/OFF et les proportions de bande passante à chaque slice."""
        for i, port in enumerate(self.ports):
            rates = []
            for s in range(self.config.num_slices):
                if c_final[s] == 1:
                    rates.append(rho[s] * self.config.max_bandwidth_mbps)
                else:
                    rates.append(0.0)
            data = {"id": port, "rates": rates}
            self._make_rest_request("/setaction", method="POST", data=data)


# ---------------------------------------------------------------------------
# Algorithme 2 : Filtre des seuils (Contrôleur SDN 1)
# ---------------------------------------------------------------------------

class SDNController1:
    """
    Étape 2 : Gestion des Seuils.
    Reçoit c_init (ON/OFF par l'agent) et les charges réelles.
    Éteint les slices dont le trafic est sous αseuil·Maxi.
    Collecte le trafic orphelin dans Vrediriger.
    """

    def __init__(self, config: NetworkConfig):
        self.config = config

    def filter_thresholds(
        self,
        c_init: np.ndarray,
        traffic_loads: np.ndarray,
        traffic_peak: Dict[int, float],
    ) -> Tuple[np.ndarray, List[Tuple[int, float]]]:
        """
        Algorithme 2 : RéacheminementTraffic (Partie 1) — Filtre des Seuils.

        Args:
            c_init:        vecteur ON/OFF proposé par l'agent PPO
            traffic_loads: charge réelle l^t_{i,b} par slice
            traffic_peak:  pic historique Maxi par slice

        Returns:
            c_filtre:     vecteur ON/OFF après filtrage
            v_rediriger:  liste (slice_id, charge) à rediriger vers EcoSlice
        """
        alpha = self.config.alpha_seuil
        eco1_id, eco2_id = self.config.eco_slice_ids

        c_filtre = c_init.copy()
        v_rediriger: List[Tuple[int, float]] = []

        for i in range(self.config.num_slices):
            # Les EcoSlices ne sont jamais filtrés ici
            if i in (eco1_id, eco2_id):
                continue

            load = traffic_loads[i]
            max_i = max(traffic_peak.get(i, 1.0), 1e-6)

            if (c_init[i] == 0) or (load < alpha * max_i):
                # Éteindre ce slice et récupérer son trafic
                c_filtre[i] = 0
                if load > 0:
                    v_rediriger.append((i, load))
                    traffic_loads[i] = 0.0  # vider le slice source
            else:
                c_filtre[i] = 1

        return c_filtre, v_rediriger


# ---------------------------------------------------------------------------
# Algorithme 3 : Routage et Débordement (Contrôleur SDN 2)
# ---------------------------------------------------------------------------

class SDNController2:
    """
    Étape 3 : Priorité et EcoSlice Bis.
    Trie le trafic orphelin par priorité, le verse dans EcoSlice1,
    active EcoSlice2 si nécessaire, lève Surcharge si les deux débordent.

    Priority order (index): URLLC(0) > URLLC_eMBB_MIX(1) > mMTC(3) > eMBB(2)
    """

    def __init__(self, config: NetworkConfig):
        self.config = config
        # Build priority lookup: slice_index → rank (lower = higher priority)
        self._priority_rank: Dict[int, int] = {
            idx: rank for rank, idx in enumerate(config.slice_priority)
        }

    def route_and_overflow(
        self,
        v_rediriger: List[Tuple[int, float]],
        load_eco1: float,
        load_eco2: float,
    ) -> Tuple[float, float, bool, bool]:
        """
        Algorithme 3 : RéacheminementTraffic (Partie 2) — Routage et Débordement.

        Args:
            v_rediriger: liste (slice_id, charge) à router
            load_eco1:   charge courante EcoSlice1
            load_eco2:   charge courante EcoSlice2

        Returns:
            load_eco1, load_eco2, eco2_active, surcharge
        """
        C_eco1 = self.config.eco1_capacity_mbps
        C_eco2 = self.config.eco2_capacity_mbps
        seuil75 = self.config.eco1_overflow_threshold * C_eco1

        eco2_active = False
        surcharge = False

        # Trier par priorité décroissante selon config.slice_priority
        def priority_key(item):
            slice_id, _ = item
            return self._priority_rank.get(slice_id, len(self._priority_rank))

        v_sorted = sorted(v_rediriger, key=priority_key)

        # Verser chaque flux dans EcoSlice1
        for _, load_i in v_sorted:
            load_eco1 += load_i

        # Si EcoSlice1 dépasse 75%, activer EcoSlice2 et transférer l'excédent
        if load_eco1 > seuil75:
            excess = load_eco1 - seuil75
            load_eco2 += excess
            load_eco1 = seuil75
            eco2_active = True

        # Vérifier la saturation totale
        if eco2_active and load_eco2 > C_eco2:
            surcharge = True

        return load_eco1, load_eco2, eco2_active, surcharge


# ---------------------------------------------------------------------------
# Algorithme 4 : Sécurité et Réallocation (Contrôleur SDN 2)
# ---------------------------------------------------------------------------

class SafetyReallocator:
    """
    Étape 4 : Réallocation et Sécurité.
    En cas de Surcharge, rallume la slice avec le trafic le plus lourd.
    Calcule rho (proportions BW), ΔE (gain énergétique), L (perte paquets), η (QoS).
    """

    def __init__(self, config: NetworkConfig):
        self.config = config

    def reallocate(
        self,
        c_filtre: np.ndarray,
        v_rediriger: List[Tuple[int, float]],
        traffic_loads: np.ndarray,
        surcharge: bool,
        load_eco1: float,
        load_eco2: float,
        f_base: float,
        f_current: float,
        qos_per_slice: np.ndarray,
    ) -> TrafficRedirectResult:
        """
        Algorithme 4 : RéacheminementTraffic (Partie 3) — Sécurité et Réallocation.

        Args:
            c_filtre:       vecteur ON/OFF après filtre seuils
            v_rediriger:    trafic orphelin
            traffic_loads:  charge réelle par slice l^t_{i,b}
            surcharge:      alarme de saturation (Algo 3)
            load_eco1/2:    charges EcoSlices après routage
            f_base:         énergie référence (toutes slices allumées)
            f_current:      énergie courante
            qos_per_slice:  taux de satisfaction QoS par slice (0..1)

        Returns:
            TrafficRedirectResult avec c_final, rho, delta_E, L, eta_b
        """
        result = TrafficRedirectResult()
        result.c_final = c_filtre.copy()
        result.load_eco1 = load_eco1
        result.load_eco2 = load_eco2
        result.surcharge = surcharge
        result.v_rediriger = v_rediriger

        # --- Sécurité : rallumer le slice le plus chargé si Surcharge ---
        if surcharge and v_rediriger:
            # Trouver le flux le plus volumineux dans Vrediriger
            heaviest_id, heaviest_load = max(v_rediriger, key=lambda x: x[1])
            result.c_final[heaviest_id] = 1
            traffic_loads[heaviest_id] = heaviest_load   # remettre sa charge
            result.load_eco1 = max(0.0, load_eco1 - heaviest_load)

        # --- Calcul des proportions de bande passante ρ ---
        active_mask = result.c_final == 1
        total_active_load = np.sum(traffic_loads[active_mask])
        rho = np.zeros(self.config.num_slices)
        if total_active_load > 0:
            for i in range(self.config.num_slices):
                if result.c_final[i] == 1:
                    rho[i] = traffic_loads[i] / total_active_load
        result.rho = rho

        # --- Gain énergétique ΔE ---
        if f_base and f_base > 0:
            result.delta_E = (f_base - f_current) / f_base
        else:
            result.delta_E = 0.0

        # --- Taux de perte de paquets L ---
        if surcharge:
            eco_overflow = (result.load_eco1 + result.load_eco2) \
                           - (self.config.eco1_capacity_mbps + self.config.eco2_capacity_mbps)
            eco_total = result.load_eco1 + result.load_eco2
            result.L = max(0.0, eco_overflow / max(eco_total, 1e-6))
        else:
            result.L = 0.0

        # --- Taux de satisfaction QoS moyen η_b ---
        n_slices = self.config.num_slices
        result.eta_b = float(np.mean(qos_per_slice)) if len(qos_per_slice) == n_slices else 0.0

        return result


# ---------------------------------------------------------------------------
# Réseaux Actor / Critic PPO (inchangés architecturalement)
# ---------------------------------------------------------------------------

class Actor(nn.Module):
    """
    Produit les paramètres alpha d'une distribution Dirichlet pour les proportions de BW
    ET les logits d'une distribution de Bernoulli pour les décisions ON/OFF par slice.
    """
    def __init__(self, state_size: int, num_slices: int, hidden_size: int = 256):
        super().__init__()
        self.num_slices = num_slices
        self.net = nn.Sequential(
            nn.Linear(state_size, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, hidden_size), nn.ReLU(),
        )
        # Tête Dirichlet pour l'allocation de BW
        self.bw_head = nn.Linear(hidden_size, num_slices)
        # Tête Bernoulli pour ON/OFF par slice
        self.onoff_head = nn.Linear(hidden_size, num_slices)

    def forward(self, x: torch.Tensor):
        h = self.net(x)
        alpha = F.softplus(self.bw_head(h)) + 1e-8       # Dirichlet concentrations
        onoff_logits = self.onoff_head(h)                  # Bernoulli logits
        return alpha, onoff_logits


class Critic(nn.Module):
    def __init__(self, state_size: int, hidden_size: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_size, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# Agent PPO centralisé — Architecture à double contrôleur
# ---------------------------------------------------------------------------

class CentralPPOAgent:
    def __init__(self, config: NetworkConfig):
        self.config = config
        self.state_size = config.num_ports_per_switch * config.num_slices * 4
        self.num_slices = config.num_slices

        self.actor  = Actor(self.state_size, self.num_slices)
        self.critic = Critic(self.state_size)
        self.actor_opt  = optim.Adam(self.actor.parameters(),  lr=3e-4)
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=3e-4)

        self.gamma    = 0.99
        self.clip_eps = 0.2
        self.lmbda    = 0.95

        self.episode_rewards: List[float] = []

        # Sous-systèmes
        self.env        = SDNEnvironment(config)
        self.ctrl1      = SDNController1(config)
        self.ctrl2      = SDNController2(config)
        self.safety     = SafetyReallocator(config)
        self.forecast_provider: Optional[ForecastDemandProvider] = None
        if config.forecast_csv:
            self.forecast_provider = ForecastDemandProvider(
                csv_path=config.forecast_csv,
                slice_names=config.slice_names,
                quantile=config.forecast_quantile,
                loop=config.forecast_loop,
            )
            logger.info(
                "Prévisions chargées depuis %s | quantile=%s | mode=%s | points=%d",
                config.forecast_csv,
                config.forecast_quantile,
                config.forecast_mode,
                len(self.forecast_provider.loads),
            )

    # ------------------------------------------------------------------
    # Algorithme 1 : Décision d'activation brute (Agent PPO)
    # ------------------------------------------------------------------

    def select_action(self, state: np.ndarray):
        """
        Algorithme 1 : échantillonne l'action (ON/OFF + BW) depuis πθ(·|s).

        Returns:
            bw_action:    proportions BW (vecteur Dirichlet, somme=1)
            c_init:       vecteur binaire ON/OFF par slice
            logp:         log-probabilité jointe (BW + ON/OFF)
            bw_dist:      distribution Dirichlet (pour entropy)
            onoff_dist:   distribution Bernoulli  (pour entropy)
        """
        st = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        alpha, onoff_logits = self.actor(st)

        bw_dist    = Dirichlet(alpha)
        onoff_dist = Bernoulli(logits=onoff_logits)

        bw_action = bw_dist.sample().squeeze(0)           # [num_slices]
        c_raw     = onoff_dist.sample().squeeze(0)        # [num_slices], float 0/1

        # EcoSlice1 (idx 0) toujours ON (Algo 1, ligne 4)
        c_raw[self.config.eco_slice_ids[0]] = 1.0

        c_init = c_raw.detach().numpy().astype(int)

        logp_bw    = bw_dist.log_prob(bw_action)
        logp_onoff = onoff_dist.log_prob(c_raw).sum()
        logp       = logp_bw + logp_onoff

        return bw_action, c_init, logp, bw_dist, onoff_dist

    # ------------------------------------------------------------------
    # Appel complet de RéacheminementTraffic (Algos 2 → 3 → 4)
    # ------------------------------------------------------------------

    def redirect_traffic(
        self,
        c_init: np.ndarray,
        traffic_loads: np.ndarray,
    ) -> TrafficRedirectResult:
        """
        Exécute la chaîne complète :
          Algo 2 → Algo 3 → Algo 4
        en retournant un TrafficRedirectResult.
        """
        # Énergie de référence (calculée une seule fois)
        f_current = self.env.get_energy_consumption()
        if self.env.f_base is None:
            self.env.f_base = f_current if f_current > 0 else 1.0

        # ---------- Algo 2 : Filtre des Seuils ----------
        c_filtre, v_rediriger = self.ctrl1.filter_thresholds(
            c_init,
            traffic_loads.copy(),
            self.env.traffic_peak,
        )

        # Charges initiales des EcoSlices
        eco1_id = self.config.eco_slice_ids[0]
        eco2_id = self.config.eco_slice_ids[1]
        load_eco1 = float(traffic_loads[eco1_id])
        load_eco2 = float(traffic_loads[eco2_id])

        # ---------- Algo 3 : Routage et Débordement ----------
        load_eco1, load_eco2, eco2_active, surcharge = self.ctrl2.route_and_overflow(
            v_rediriger, load_eco1, load_eco2
        )

        # ---------- Algo 4 : Sécurité et Réallocation ----------
        state = self.env.get_state()
        # QoS par slice : index 2, 6, 10, … (latency score dans l'état)
        qos_per_slice = state[2::4][:self.num_slices]

        result = self.safety.reallocate(
            c_filtre       = c_filtre,
            v_rediriger    = v_rediriger,
            traffic_loads  = traffic_loads.copy(),
            surcharge      = surcharge,
            load_eco1      = load_eco1,
            load_eco2      = load_eco2,
            f_base         = self.env.f_base,
            f_current      = f_current,
            qos_per_slice  = qos_per_slice,
        )
        result.eco2_active = eco2_active

        return result

    # ------------------------------------------------------------------
    # Calcul de la récompense
    # ------------------------------------------------------------------

    def compute_reward(self, next_state: np.ndarray, result: TrafficRedirectResult) -> float:
        """
        Récompense combinant :
          - satisfaction QoS (latence, débit) pondérée par priorité de slice
          - perte de paquets (pénalité forte)
          - gain énergétique (Algo 4 ΔE)
          - taux de satisfaction QoS moyen η_b (Algo 4)

        State layout (per port per slice): [drop_rate, bw_util, latency_score, tp_util]
        4 slices × 4 features = 16 values per port.
        Priority weights: URLLC(0)=1.0, MIX(1)=0.8, eMBB(2)=0.5, mMTC(3)=0.3
        """
        num_ports = max(self.env.numports, 1)
        n = self.num_slices  # 4

        # Slice priority weights (index 0..3)
        prio_weights = np.array([1.0, 0.8, 0.5, 0.3], dtype=np.float32)

        drop_scores     = next_state[0::4][:n]    # drop_rate per slice (first port avg)
        latency_scores  = next_state[2::4][:n]    # (1 - lat/max_lat) per slice
        tp_util         = next_state[3::4][:n]    # throughput/rate per slice

        # Weighted QoS terms
        r_latency    = float(np.dot(prio_weights, latency_scores) / prio_weights.sum())
        r_drop       = float(np.dot(prio_weights, 1 - drop_scores) / prio_weights.sum())
        r_throughput = float(np.dot(prio_weights, tp_util) / prio_weights.sum())

        # Penalties / bonuses from Algo 4
        r_loss   = -2.0 * result.L
        r_energy =  0.5 * result.delta_E
        r_qos_b  =  0.5 * result.eta_b

        reward = (
            0.7 * r_latency
            + 0.4 * r_drop
            + 0.4 * r_throughput
            + r_loss
            + r_energy
            + r_qos_b
        ) / (num_ports * n)

        return float(reward)

    # ------------------------------------------------------------------
    # Mise à jour PPO
    # ------------------------------------------------------------------

    def compute_gae(self, rewards: List[float], values: List[float]) -> List[float]:
        values = values + [0.0]
        gae = 0.0
        adv = []
        for i in reversed(range(len(rewards))):
            delta = rewards[i] + self.gamma * values[i + 1] - values[i]
            gae   = delta + self.gamma * self.lmbda * gae
            adv.insert(0, gae)
        return adv

    def update(self, traj: dict) -> None:
        states     = torch.tensor(np.array(traj['states']),    dtype=torch.float32)
        bw_actions = torch.tensor(np.array(traj['bw_actions']), dtype=torch.float32)
        on_actions = torch.tensor(np.array(traj['on_actions']), dtype=torch.float32)
        old_logp   = torch.stack(traj['logps'])
        returns    = torch.tensor(traj['returns'],              dtype=torch.float32)
        adv        = torch.tensor(
            self.compute_gae(traj['rewards'], traj['values']), dtype=torch.float32
        )
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        for _ in range(4):
            alpha, onoff_logits = self.actor(states)
            bw_dist    = Dirichlet(alpha)
            onoff_dist = Bernoulli(logits=onoff_logits)

            logp_bw    = bw_dist.log_prob(bw_actions)
            logp_onoff = onoff_dist.log_prob(on_actions).sum(dim=-1)
            logp       = logp_bw + logp_onoff

            ratio  = torch.exp(logp - old_logp)
            surr1  = ratio * adv
            surr2  = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * adv

            entropy = bw_dist.entropy().mean() + onoff_dist.entropy().mean()
            actor_loss  = -torch.min(surr1, surr2).mean() - 0.01 * entropy
            critic_loss = F.mse_loss(self.critic(states).squeeze(1), returns)

            self.actor_opt.zero_grad()
            actor_loss.backward(retain_graph=True)
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
            self.actor_opt.step()

            self.critic_opt.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
            self.critic_opt.step()

    # ------------------------------------------------------------------
    # Visualisation
    # ------------------------------------------------------------------

    def plot_rewards(self, x, y, alpha: float = 0.1) -> None:
        def ema(data, a):
            s = [data[0]]
            for v in data[1:]:
                s.append(a * v + (1 - a) * s[-1])
            return s

        plt.figure()
        plt.plot(list(x), ema(y, alpha), label=f'EMA (alpha={alpha})', color='blue')
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.title('Reward per Episode (Central PPO – Dual Controller – 4 Slices)')
        plt.legend()
        plt.savefig("reward_central_dual_ctrl.png")
        plt.show()

    # ------------------------------------------------------------------
    # Boucle d'apprentissage principale
    # ------------------------------------------------------------------

    def learn(self, episodes: int = 100, steps_per_ep: int = 100) -> None:
        """
        Boucle principale implémentant Algorithme 1 complet :
          Pour chaque t :
            1. Agent PPO → c_init, bw_action
            2. RéacheminementTraffic (Algos 2→3→4) → c_final, rho, ΔE, L, η
            3. Application de c_final / rho via le SDN
            4. Collecte de la récompense et mise à jour PPO
        """
        for ep in range(episodes):
            state = self.env.get_state()
            traj = {
                'states': [], 'bw_actions': [], 'on_actions': [],
                'logps': [], 'rewards': [], 'values': [], 'returns': [],
            }
            ep_reward = 0.0

            for t in range(steps_per_ep):
                # ------ Algo 1 : décision PPO ------
                bw_action, c_init, logp, _, _ = self.select_action(state)
                val = self.critic(torch.tensor(state, dtype=torch.float32)).item()

                # ------ Algos 2→3→4 : RéacheminementTraffic ------
                observed_loads = self.env.get_traffic_loads()
                traffic_loads = observed_loads
                forecast_loads = None
                forecast_timestamp = None
                if self.forecast_provider is not None:
                    traffic_loads, forecast_loads = self.forecast_provider.combine(
                        observed_loads,
                        self.config.forecast_mode,
                    )
                    forecast_timestamp = self.forecast_provider.current_timestamp
                result = self.redirect_traffic(c_init, traffic_loads)

                # ------ Application SDN ------
                self.env.apply_activation(result.c_final, result.rho)
                time.sleep(self.config.observation_interval)

                next_state = self.env.get_state()

                # ------ Récompense ------
                reward = self.compute_reward(next_state, result)

                traj['states'].append(state)
                traj['bw_actions'].append(bw_action.detach().numpy())
                traj['on_actions'].append(c_init.astype(np.float32))
                traj['logps'].append(logp.detach())
                traj['values'].append(val)
                traj['rewards'].append(reward)

                state = next_state
                ep_reward += reward

                forecast_msg = ""
                if forecast_loads is not None:
                    forecast_msg = (
                        f" | forecast_ts={forecast_timestamp}"
                        f" | observed={np.round(observed_loads, 3)}"
                        f" | forecast={np.round(forecast_loads, 3)}"
                    )
                logger.info(
                    f"Ep {ep} | Step {t} | c_final={result.c_final} "
                    f"| rho={np.round(result.rho, 3)} | ΔE={result.delta_E:.3f} "
                    f"| L={result.L:.4f} | η={result.eta_b:.3f} "
                    f"| surcharge={result.surcharge}{forecast_msg} | reward={reward:.4f}"
                )

            # Calcul des retours
            R = 0.0
            returns = []
            for r in reversed(traj['rewards']):
                R = r + self.gamma * R
                returns.insert(0, R)
            traj['returns'] = returns

            self.update(traj)

            avg_reward = float(np.mean(traj['rewards']))
            self.episode_rewards.append(avg_reward)
            logger.info(f"=== Episode {ep} | Total Reward: {ep_reward:.2f} | Avg: {avg_reward:.4f} ===")

        self.plot_rewards(range(episodes), self.episode_rewards)


# ---------------------------------------------------------------------------
# Point d'entrée
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    cfg   = NetworkConfig()
    agent = CentralPPOAgent(cfg)
    agent.learn(episodes=100, steps_per_ep=100)
