"""
traffic_generator.py
--------------------
Synthetic traffic generation for simulation experiments.
Supports CBR and Poisson (bursty) traffic as used in Section 7.3.
"""

import numpy as np
import pandas as pd
import os


class TrafficGenerator:
    """
    Generate synthetic network traffic traces.

    Parameters
    ----------
    traffic_type : str
        'cbr' — Constant Bit Rate
        'poisson' — Poisson bursty traffic
    cbr_rate_mbps : float
    poisson_lambda : float
        Packets per second
    packet_size_min : int
        Bytes
    packet_size_max : int
        Bytes
    duration_seconds : int
    seed : int
    """

    def __init__(self,
                 traffic_type: str = 'cbr',
                 cbr_rate_mbps: float = 5.0,
                 poisson_lambda: float = 100.0,
                 packet_size_min: int = 256,
                 packet_size_max: int = 1024,
                 duration_seconds: int = 300,
                 seed: int = 42):

        self.traffic_type     = traffic_type
        self.cbr_rate_mbps    = cbr_rate_mbps
        self.poisson_lambda   = poisson_lambda
        self.packet_size_min  = packet_size_min
        self.packet_size_max  = packet_size_max
        self.duration_seconds = duration_seconds
        self.rng              = np.random.RandomState(seed)

    def generate(self) -> pd.DataFrame:
        """
        Generate traffic trace as DataFrame.

        Returns
        -------
        pd.DataFrame with columns:
            time, packet_size, inter_arrival, flow_id
        """
        if self.traffic_type == 'cbr':
            return self._generate_cbr()
        elif self.traffic_type == 'poisson':
            return self._generate_poisson()
        else:
            raise ValueError(f"Unknown traffic type: {self.traffic_type}")

    def _generate_cbr(self) -> pd.DataFrame:
        """Constant Bit Rate traffic."""
        bits_per_pkt = self.rng.randint(
            self.packet_size_min * 8, self.packet_size_max * 8
        )
        pps          = (self.cbr_rate_mbps * 1e6) / bits_per_pkt
        inter_arrival = 1.0 / pps

        n_packets = int(self.duration_seconds * pps)
        times     = np.arange(n_packets) * inter_arrival
        sizes     = np.full(n_packets, bits_per_pkt // 8)

        return pd.DataFrame({
            'time':          times,
            'packet_size':   sizes,
            'inter_arrival': inter_arrival,
            'flow_id':       0,
        })

    def _generate_poisson(self) -> pd.DataFrame:
        """Poisson-distributed bursty traffic."""
        inter_arrivals = self.rng.exponential(
            1.0 / self.poisson_lambda, size=int(self.duration_seconds * self.poisson_lambda * 2)
        )
        times = np.cumsum(inter_arrivals)
        times = times[times < self.duration_seconds]

        sizes = self.rng.randint(
            self.packet_size_min, self.packet_size_max + 1, size=len(times)
        )

        return pd.DataFrame({
            'time':          times,
            'packet_size':   sizes,
            'inter_arrival': inter_arrivals[:len(times)],
            'flow_id':       0,
        })

    def save(self, output_dir: str = 'datasets/synthetic/'):
        """Save generated traffic trace to CSV."""
        os.makedirs(output_dir, exist_ok=True)
        df    = self.generate()
        fname = f"{output_dir}/traffic_{self.traffic_type}_{len(df)}pkts.csv"
        df.to_csv(fname, index=False)
        print(f"Traffic trace saved: {fname}  ({len(df)} packets)")
        return fname


def generate_all_traces(output_dir: str = 'datasets/synthetic/'):
    """Generate all traffic types used in paper experiments."""
    configs = [
        {'traffic_type': 'cbr',     'cbr_rate_mbps': 5.0,  'seed': 42},
        {'traffic_type': 'cbr',     'cbr_rate_mbps': 10.0, 'seed': 43},
        {'traffic_type': 'poisson', 'poisson_lambda': 100,  'seed': 44},
        {'traffic_type': 'poisson', 'poisson_lambda': 200,  'seed': 45},
    ]
    for cfg in configs:
        gen = TrafficGenerator(**cfg)
        gen.save(output_dir)


if __name__ == '__main__':
    generate_all_traces()
    print("All synthetic traffic traces generated.")
