"""
CrowdSafe Synthetic Simulation — Helbing Social Force Model (SFM)
Pure NumPy implementation of pedestrian dynamics:
- Destination attraction forces
- Pedestrian-pedestrian repulsive and compression forces
- Pedestrian-wall and obstacle repulsion forces
- Bottleneck arching and panic crush dynamics
"""

import math
from typing import List, Dict, Tuple, Optional
import numpy as np


class PedestrianAgent:
    """Represents a single humanoid agent in the 2D simulation space."""

    def __init__(
        self,
        agent_id: int,
        start_pos: Tuple[float, float],
        route: List[str],
        desired_speed: float = 1.25,
        radius: float = 0.25,
        mass: float = 75.0,
        lateral_offset: float = 0.0,
        height: float = 1.72,
        color_idx: int = 0,
    ):
        self.id = agent_id
        self.pos = np.array(start_pos, dtype=np.float64)
        self.vel = np.zeros(2, dtype=np.float64)
        self.route = route
        self.route_idx = 0
        self.desired_speed = desired_speed
        self.max_speed = desired_speed * 1.4
        self.radius = radius
        self.mass = mass
        self.lateral_offset = lateral_offset
        self.height = height
        self.color_idx = color_idx
        
        # State tracking
        self.yaw_radians = 0.0
        self.is_active = True
        self.reached_goal = False
        self.gait_phase = np.random.uniform(0, 2 * math.pi)

    def get_current_waypoint(self, nav_nodes: Dict[str, Tuple[float, float]]) -> Optional[np.ndarray]:
        if self.route_idx >= len(self.route):
            return None
        node_id = self.route[self.route_idx]
        base_pos = np.array(nav_nodes[node_id], dtype=np.float64)
        # Apply deterministic lateral offset orthogonal to corridor
        offset = np.array([0.0, self.lateral_offset], dtype=np.float64)
        return base_pos + offset


class SocialForceEngine:
    """
    Vectorized Social Force Model engine for dense crowd navigation.
    Implements Helbing & Molnár (1995) formulation with contact forces.
    """

    def __init__(
        self,
        nav_nodes: Dict[str, Tuple[float, float]],
        walls: List[Dict[str, any]],
        obstacles: Optional[List[Dict[str, any]]] = None,
        tau: float = 0.5,           # Relaxation time (seconds)
        A_ped: float = 2000.0,      # Pedestrian repulsion strength (N)
        B_ped: float = 0.08,        # Pedestrian repulsion range (m)
        k_ped: float = 1.2e5,       # Compression body elasticity (kg/s^2)
        kappa_ped: float = 2.4e5,   # Sliding friction coefficient (kg/m s)
        A_wall: float = 2200.0,     # Wall repulsion strength (N)
        B_wall: float = 0.10,       # Wall repulsion range (m)
        k_wall: float = 1.5e5,      # Wall hard contact stiffness
    ):
        self.nav_nodes = {k: tuple(v) for k, v in nav_nodes.items()}
        self.walls = walls
        self.obstacles = obstacles or []
        
        # Physical constants
        self.tau = tau
        self.A_ped = A_ped
        self.B_ped = B_ped
        self.k_ped = k_ped
        self.kappa_ped = kappa_ped
        self.A_wall = A_wall
        self.B_wall = B_wall
        self.k_wall = k_wall

        # Pre-process wall segments as NumPy arrays [x1, y1, x2, y2]
        self._wall_segments = []
        for w in self.walls:
            p1 = np.array(w["start"], dtype=np.float64)
            p2 = np.array(w["end"], dtype=np.float64)
            self._wall_segments.append((p1, p2))

        # Pre-process obstacles (circular pillars and boxes)
        self._cyl_obstacles = []
        for obs in self.obstacles:
            if obs.get("type") == "cylinder":
                cx, cy = obs["position"][0], obs["position"][1]
                radius = obs["radius"]
                self._cyl_obstacles.append((np.array([cx, cy], dtype=np.float64), radius))

        self.agents: List[PedestrianAgent] = []

    def add_agent(self, agent: PedestrianAgent):
        self.agents.append(agent)

    def step(self, dt: float = 0.04):
        """Advances the simulation by dt seconds (default 25 FPS = 0.04s)."""
        active_agents = [a for a in self.agents if a.is_active and not a.reached_goal]
        num_agents = len(active_agents)
        if num_agents == 0:
            return

        # Extract state matrices for vectorized operations
        positions = np.array([a.pos for a in active_agents], dtype=np.float64)   # (N, 2)
        velocities = np.array([a.vel for a in active_agents], dtype=np.float64)  # (N, 2)
        desired_speeds = np.array([a.desired_speed for a in active_agents])      # (N,)
        max_speeds = np.array([a.max_speed for a in active_agents])              # (N,)
        radii = np.array([a.radius for a in active_agents])                      # (N,)
        masses = np.array([a.mass for a in active_agents])                        # (N,)

        # -------------------------------------------------------------------------
        # 1. Target Attraction Force (Desire to reach current waypoint)
        # -------------------------------------------------------------------------
        f_desire = np.zeros_like(positions)
        for i, agent in enumerate(active_agents):
            target = agent.get_current_waypoint(self.nav_nodes)
            if target is None:
                continue

            diff = target - positions[i]
            dist = np.linalg.norm(diff)
            
            # Check if waypoint is reached
            if dist < 0.8:
                agent.route_idx += 1
                new_target = agent.get_current_waypoint(self.nav_nodes)
                if new_target is None:
                    agent.reached_goal = True
                    continue
                diff = new_target - positions[i]
                dist = np.linalg.norm(diff)

            if dist > 1e-4:
                e_dir = diff / dist
                v_des = e_dir * desired_speeds[i]
                f_desire[i] = masses[i] * (v_des - velocities[i]) / self.tau

        # -------------------------------------------------------------------------
        # 2. Vectorized Inter-Pedestrian Repulsive & Contact Forces
        # -------------------------------------------------------------------------
        f_ped = np.zeros_like(positions)
        if num_agents > 1:
            # Pairwise difference vectors: delta_pos[i, j] = pos[i] - pos[j]
            delta_pos = positions[:, np.newaxis, :] - positions[np.newaxis, :, :] # (N, N, 2)
            dist_matrix = np.linalg.norm(delta_pos, axis=2)                       # (N, N)
            
            # Mask out self-interaction
            np.fill_diagonal(dist_matrix, np.inf)

            # Combined radii: r_sum[i, j] = radii[i] + radii[j]
            r_sum = radii[:, np.newaxis] + radii[np.newaxis, :]                  # (N, N)
            
            # Unit direction vectors from j to i
            safe_dist = np.where(dist_matrix < 1e-4, 1e-4, dist_matrix)
            n_ij = delta_pos / safe_dist[:, :, np.newaxis]                       # (N, N, 2)
            
            # Social distance: r_ij - d_ij
            overlap = r_sum - dist_matrix
            np.fill_diagonal(overlap, 0.0)
            
            # Helbing exponential social repulsive force
            f_social_mag = self.A_ped * np.exp(overlap / self.B_ped)             # (N, N)
            
            # Contact / compression forces when bodies touch (overlap > 0)
            in_contact = overlap > 0
            f_contact_mag = np.where(in_contact, self.k_ped * overlap, 0.0)      # (N, N)
            
            total_rep_mag = f_social_mag + f_contact_mag
            # Zero out diagonal
            np.fill_diagonal(total_rep_mag, 0.0)
            
            # Sum forces acting on each agent i from all neighbors j
            f_ped = np.sum(total_rep_mag[:, :, np.newaxis] * n_ij, axis=1)

            # Tangential sliding friction when in contact
            if np.any(in_contact):
                t_ij = np.stack([-n_ij[:, :, 1], n_ij[:, :, 0]], axis=2) # Orthogonal tangent
                delta_vel = velocities[np.newaxis, :, :] - velocities[:, np.newaxis, :]
                tangent_vel = np.sum(delta_vel * t_ij, axis=2)
                f_friction_mag = np.where(in_contact, self.kappa_ped * overlap * tangent_vel, 0.0)
                np.fill_diagonal(f_friction_mag, 0.0)
                f_friction = np.sum(f_friction_mag[:, :, np.newaxis] * t_ij, axis=1)
                f_ped += f_friction

        # -------------------------------------------------------------------------
        # 3. Wall and Obstacle Repulsion Forces
        # -------------------------------------------------------------------------
        f_wall = np.zeros_like(positions)
        for i in range(num_agents):
            pos_i = positions[i]
            r_i = radii[i]
            
            # Segment-to-point distance check for each wall
            for p1, p2 in self._wall_segments:
                seg_vec = p2 - p1
                seg_len_sq = np.dot(seg_vec, seg_vec)
                if seg_len_sq < 1e-8:
                    continue
                
                # Projection factor t clamped to segment [0, 1]
                t = np.clip(np.dot(pos_i - p1, seg_vec) / seg_len_sq, 0.0, 1.0)
                closest = p1 + t * seg_vec
                diff = pos_i - closest
                dist = np.linalg.norm(diff)
                
                if dist < 1e-4:
                    n_w = np.array([0.0, 1.0])
                    dist = 1e-4
                else:
                    n_w = diff / dist

                wall_overlap = r_i - dist
                # Exponential repulsion
                f_w_mag = self.A_wall * math.exp(wall_overlap / self.B_wall)
                # Hard contact if clipping
                if wall_overlap > 0:
                    f_w_mag += self.k_wall * wall_overlap
                
                f_wall[i] += f_w_mag * n_w

            # Circular obstacle repulsion (pillars)
            for center, obs_radius in self._cyl_obstacles:
                diff = pos_i - center
                dist = np.linalg.norm(diff)
                if dist < 1e-4:
                    continue
                n_obs = diff / dist
                eff_overlap = (r_i + obs_radius) - dist
                f_obs_mag = self.A_wall * math.exp(eff_overlap / self.B_wall)
                if eff_overlap > 0:
                    f_obs_mag += self.k_wall * eff_overlap
                f_wall[i] += f_obs_mag * n_obs

        # -------------------------------------------------------------------------
        # 4. Numerical Integration (Euler / Velocity Verlet)
        # -------------------------------------------------------------------------
        total_forces = f_desire + f_ped + f_wall
        accelerations = total_forces / masses[:, np.newaxis]
        
        # Update velocities
        new_velocities = velocities + accelerations * dt
        
        # Clamp speed to physical max
        speeds = np.linalg.norm(new_velocities, axis=1)
        speed_clamp = np.where(speeds > max_speeds, max_speeds / np.maximum(speeds, 1e-4), 1.0)
        new_velocities *= speed_clamp[:, np.newaxis]
        
        # Update positions
        new_positions = positions + new_velocities * dt

        # Apply back to agent objects and calculate dynamic orientation & gait
        for idx, agent in enumerate(active_agents):
            agent.pos = new_positions[idx]
            agent.vel = new_velocities[idx]
            
            # Yaw orientation (face direction of movement if moving)
            current_speed = speeds[idx]
            if current_speed > 0.05:
                agent.yaw_radians = math.atan2(new_velocities[idx, 1], new_velocities[idx, 0])
                # Advance gait phase
                stride_freq = current_speed / 0.70  # ~0.7m average stride
                agent.gait_phase += 2 * math.pi * stride_freq * dt
            
            # Clamp positions to mall bounding box [0.2, 49.8] x [0.2, 29.8]
            agent.pos[0] = max(0.2, min(49.8, agent.pos[0]))
            agent.pos[1] = max(0.2, min(29.8, agent.pos[1]))

    def get_agent_states(self) -> List[Dict[str, any]]:
        """Returns snapshot of all agent coordinates, velocities, and orientations."""
        states = []
        for a in self.agents:
            # Vertical gait bobbing oscillation
            bob_offset = 0.025 * abs(math.sin(a.gait_phase))
            states.append({
                "id": a.id,
                "x": round(float(a.pos[0]), 3),
                "y": round(float(a.pos[1]), 3),
                "z": round(a.height + bob_offset, 3),
                "vx": round(float(a.vel[0]), 3),
                "vy": round(float(a.vel[1]), 3),
                "speed": round(float(np.linalg.norm(a.vel)), 3),
                "yaw": round(float(a.yaw_radians), 4),
                "height": a.height,
                "color_idx": a.color_idx,
                "is_active": a.is_active and not a.reached_goal
            })
        return states
