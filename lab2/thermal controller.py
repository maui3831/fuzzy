import pygame
import random
import math
from collections import deque

# --- Main Application Class ---
class FuzzyThermalControl:
    """
    A Pygame application that simulates a fuzzy logic-based thermal control system.
    This class handles the simulation logic, fuzzy controller, and graphical user interface.
    """
    def __init__(self):
        # 1. Pygame Initialization
        pygame.init()
        pygame.display.set_caption("Fuzzy Logic Thermal Control System")

        # Screen and Layout Dimensions
        self.WIDTH, self.HEIGHT = 1280, 720
        self.screen = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        
        # Load fonts for rendering text
        self.font_big = pygame.font.Font(None, 48)
        self.font_medium = pygame.font.Font(None, 32)
        self.font_small = pygame.font.Font(None, 24)
        self.font_mono = pygame.font.SysFont('consolas', 22)
        self.font_tiny = pygame.font.SysFont('consolas', 16)

        # Color palette for the UI
        self.COLOR = {
            'bg': (240, 245, 255),
            'panel': (255, 255, 255),
            'shadow': (200, 200, 220),
            'text': (55, 65, 81),
            'text_light': (107, 114, 128),
            'accent_blue': (37, 99, 235),
            'accent_red': (220, 38, 38),
            'accent_green': (22, 163, 74),
            'accent_purple': (147, 51, 234),
            'accent_orange': (234, 88, 12),
            'grid': (229, 231, 235),
            'btn_start': (22, 163, 74),
            'btn_start_hover': (21, 128, 61),
            'btn_pause': (220, 38, 38),
            'btn_pause_hover': (185, 28, 28),
            'btn_reset': (75, 85, 99),
            'btn_reset_hover': (55, 65, 81),
        }

        # 2. State Variables
        self.current_temp = random.uniform(10.0, 40.0)
        self.target_temp = 25.0
        self.is_running = False
        self.temp_history = deque(maxlen=120)  # Max 120 points (60s of data)
        self.error = 0.0
        self.error_dot = 0.0
        self.control_action = 'NEUTRAL'
        self.action_strength = 0.0
        self.time_step = 0
        self.prev_error = 0.0
        self.last_update_time = 0
        self.ambient_temp = 30.0
        self.ambient_mode = 'Summer'
        
        # Counter for consecutive neutral actions
        self.neutral_action_count = 0
        # Flag to track ambient drift application
        self.is_applying_ambient_drift = False

        # UI Element Rectangles for layout and interaction
        self._initialize_layout()

    def _initialize_layout(self):
        """Define the Rect objects for all UI panels to structure the layout."""
        self.graph_panel_rect = pygame.Rect(30, 80, 820, 360)
        self.thermo_panel_rect = pygame.Rect(30, 460, 820, 230)
        self.control_panel_rect = pygame.Rect(870, 80, 380, 200)
        self.ambient_panel_rect = pygame.Rect(870, 290, 380, 70)
        self.metrics_panel_rect = pygame.Rect(870, 370, 380, 230)
        self.rules_panel_rect = pygame.Rect(870, 610, 380, 140)
        
        # Define Rects for interactive buttons
        self.btn_start_pause_rect = pygame.Rect(self.control_panel_rect.x + 30, self.control_panel_rect.y + 130, 150, 45)
        self.btn_reset_rect = pygame.Rect(self.control_panel_rect.x + 200, self.control_panel_rect.y + 130, 150, 45)
        self.btn_plus_rect = pygame.Rect(self.control_panel_rect.x + 290, self.control_panel_rect.y + 75, 40, 40)
        self.btn_minus_rect = pygame.Rect(self.control_panel_rect.x + 240, self.control_panel_rect.y + 75, 40, 40)
        self.btn_ambient_toggle_rect = pygame.Rect(self.ambient_panel_rect.x + 30, self.ambient_panel_rect.y + 20, self.ambient_panel_rect.width - 60, 35)

    # 3. Fuzzy Logic Engine
    def triangular_mf(self, x, a, b, c):
        """Triangular membership function."""
        if x <= a or x >= c: return 0.0
        if x == b: return 1.0
        if x < b: return (x - a) / (b - a)
        return (c - x) / (c - b)

    def trapezoidal_mf(self, x, a, b, c, d):
        """Trapezoidal membership function."""
        if x <= a or x >= d: return 0.0
        if b <= x <= c: return 1.0
        if x < b: return (x - a) / (b - a)
        return (d - x) / (d - c)

    def fuzzy_controller(self, err, err_dot):
        """
        Core fuzzy logic controller.
        Takes error and error rate as input and returns a control action.
        """
        # Fuzzification for error (-20 to 20)
        err_neg = self.trapezoidal_mf(err, -20, -20, -2, 0)
        err_zer = self.triangular_mf(err, -2, 0, 2)
        err_pos = self.trapezoidal_mf(err, 0, 2, 20, 20)

        # Fuzzification for error_dot (-2 to 2)
        err_dot_neg = self.trapezoidal_mf(err_dot, -2, -2, -0.2, 0)
        err_dot_zer = self.triangular_mf(err_dot, -0.2, 0, 0.2)
        err_dot_pos = self.trapezoidal_mf(err_dot, 0, 0.2, 2, 2)

        # Rule evaluation (Mamdani inference using min for 'AND' logic)
        rules = [
            {'condition': min(err_neg, err_dot_neg), 'action': 'COOL', 'strength': -1.0},
            {'condition': min(err_neg, err_dot_zer), 'action': 'COOL', 'strength': -0.8},
            {'condition': min(err_neg, err_dot_pos), 'action': 'COOL', 'strength': -0.6},
            {'condition': min(err_zer, err_dot_neg), 'action': 'HEAT', 'strength': 0.8},
            {'condition': min(err_zer, err_dot_zer), 'action': 'NEUTRAL','strength': 0.0},
            {'condition': min(err_zer, err_dot_pos), 'action': 'COOL', 'strength': -0.8},
            {'condition': min(err_pos, err_dot_neg), 'action': 'HEAT', 'strength': 0.6},
            {'condition': min(err_pos, err_dot_zer), 'action': 'HEAT', 'strength': 0.8},
            {'condition': min(err_pos, err_dot_pos), 'action': 'HEAT', 'strength': 1.0},
        ]

        # Defuzzification using a weighted average method
        numerator, denominator, max_condition = 0.0, 0.0, 0.0
        dominant_action = 'NEUTRAL'

        for rule in rules:
            if rule['condition'] > 0:
                numerator += rule['condition'] * rule['strength']
                denominator += rule['condition']
                if rule['condition'] > max_condition:
                    max_condition = rule['condition']
                    dominant_action = rule['action']
        
        output_strength = numerator / denominator if denominator > 0 else 0.0
        return {'action': dominant_action, 'strength': output_strength}

    # 4. Simulation Logic
    def update_simulation(self):
        """
        Updates the state of the simulation for one time step (0.5s).
        The ambient temperature drift is only applied when the system is stable
        near the target to simulate a realistic passive temperature change.
        """
        # 1. Calculate error and its rate of change
        new_error = self.target_temp - self.current_temp
        self.error_dot = (new_error - self.prev_error) / 0.5  # dt = 0.5s
        self.error = new_error
        self.prev_error = new_error

        # 2. Get control action from the fuzzy logic controller
        control = self.fuzzy_controller(self.error, self.error_dot)
        self.control_action = control['action']
        self.action_strength = control['strength']

        # 3. Initialize temperature change for this step
        temp_change = 0.0
        self.is_applying_ambient_drift = False  # Reset flag

        # 4. Apply fuzzy control action (active heating/cooling)
        if self.control_action != 'NEUTRAL' and abs(self.action_strength) > 0.01:
            distance_from_target = abs(self.error)
            control_multiplier = 0.4
            # Reduce strength when nearing target to prevent overshoot
            if distance_from_target < 2:
                control_multiplier *= 0.6
            elif distance_from_target < 5:
                control_multiplier *= 0.8
            
            temp_change += self.action_strength * control_multiplier
            # Reset the neutral counter because an action was taken
            self.neutral_action_count = 0
        else:
            # If the action is neutral, increment the counter
            self.neutral_action_count += 1

        # 5. Conditionally apply ambient temperature drift (passive change)
        is_in_target_range = abs(self.error) < 1.5  # Stable within 1.5°C
        has_been_neutral = self.neutral_action_count > 6  # After 3s of no action

        if is_in_target_range and has_been_neutral:
            drift_rate = 0.02  # Natural rate of change
            ambient_effect = drift_rate * (self.ambient_temp - self.current_temp)
            temp_change += ambient_effect
            self.is_applying_ambient_drift = True  # Set flag

        # 6. Update the current temperature
        new_temp = max(0, min(100, self.current_temp + temp_change))
        self.current_temp = new_temp

        # 7. Update history for the graph
        self.temp_history.append({
            'time': self.time_step * 0.5,
            'temp': self.current_temp,
            'target': self.target_temp,
        })
        self.time_step += 1

    def reset_simulation(self):
        """Resets the simulation to its initial state."""
        self.is_running = False
        self.current_temp = random.uniform(10.0, 40.0)
        self.temp_history.clear()
        self.error = 0.0
        self.error_dot = 0.0
        self.control_action = 'NEUTRAL'
        self.action_strength = 0.0
        self.time_step = 0
        self.prev_error = 0.0
        self.neutral_action_count = 0
        self.is_applying_ambient_drift = False
        
    # 5. GUI Drawing Methods
    def _draw_panel(self, rect, title):
        """Helper function to draw a bordered panel with a title."""
        pygame.draw.rect(self.screen, self.COLOR['shadow'], (rect.x, rect.y + 4, rect.width, rect.height), border_radius=12)
        pygame.draw.rect(self.screen, self.COLOR['panel'], rect, border_radius=12)
        
        title_surf = self.font_medium.render(title, True, self.COLOR['text'])
        self.screen.blit(title_surf, (rect.x + 20, rect.y + 15))

    def _render_and_blit_text(self, text, font, color, position, anchor="topleft"):
        """Renders text and blits it to the screen with a specific anchor."""
        surf = font.render(text, True, color)
        rect = surf.get_rect(**{anchor: position})
        self.screen.blit(surf, rect)

    def draw_graph(self):
        """Draws the real-time temperature graph."""
        panel_rect = self.graph_panel_rect
        self._draw_panel(panel_rect, "Temperature History (60s)")
        
        graph_area = pygame.Rect(panel_rect.x + 60, panel_rect.y + 60, panel_rect.width - 90, panel_rect.height - 100)
        pygame.draw.rect(self.screen, self.COLOR['bg'], graph_area)

        # Y-Axis (Temperature)
        for i in range(0, 51, 10):
            y = graph_area.bottom - (i / 50) * graph_area.height
            pygame.draw.line(self.screen, self.COLOR['grid'], (graph_area.left, y), (graph_area.right, y))
            self._render_and_blit_text(str(i), self.font_small, self.COLOR['text_light'], (graph_area.left - 10, y), anchor="midright")

        # X-Axis (Time)
        self._render_and_blit_text("Time (s)", self.font_small, self.COLOR['text_light'], (graph_area.centerx, graph_area.bottom + 25), anchor="center")
        
        if len(self.temp_history) > 1:
            points_temp, points_target = [], []
            max_time = self.temp_history[-1]['time']
            min_time = max(0, max_time - 60)

            for data in self.temp_history:
                if data['time'] >= min_time:
                    time_norm = (data['time'] - min_time) / 60
                    x = graph_area.left + time_norm * graph_area.width
                    
                    temp_norm = min(1.0, max(0.0, data['temp'] / 50.0))
                    y_temp = graph_area.bottom - temp_norm * graph_area.height
                    points_temp.append((x, y_temp))

                    target_norm = min(1.0, max(0.0, data['target'] / 50.0))
                    y_target = graph_area.bottom - target_norm * graph_area.height
                    points_target.append((x, y_target))
            
            if len(points_temp) > 1:
                pygame.draw.lines(self.screen, self.COLOR['accent_blue'], False, points_temp, 2)
            if len(points_target) > 1:
                pygame.draw.lines(self.screen, self.COLOR['accent_red'], False, points_target, 2)

    def draw_thermometer_display(self):
        """Draws the thermometer visualization and status indicators."""
        panel_rect = self.thermo_panel_rect
        self._draw_panel(panel_rect, "Live Display")
        
        section_width = panel_rect.width / 3
        thermo_center_x = panel_rect.left + section_width * 0.9
        current_center_x = panel_rect.left + section_width * 1.3
        status_center_x = panel_rect.left + section_width * 2.2

        # --- Thermometer Drawing ---
        thermo_y, bulb_radius, stem_width, stem_height = panel_rect.centery, 25, 30, 75
        
        pygame.draw.rect(self.screen, self.COLOR['grid'], (thermo_center_x - stem_width/2, thermo_y - stem_height, stem_width, stem_height), border_radius=15)
        pygame.draw.circle(self.screen, self.COLOR['grid'], (thermo_center_x, thermo_y), bulb_radius)

        temp_perc = min(1.0, max(0.0, self.current_temp / 50.0))
        fill_height = temp_perc * (stem_height)
        
        if self.current_temp < 15: fill_color = (59, 130, 246)
        elif self.current_temp < 25: fill_color = self.COLOR['accent_green']
        elif self.current_temp < 35: fill_color = (245, 158, 11)
        else: fill_color = self.COLOR['accent_red']

        pygame.draw.circle(self.screen, fill_color, (thermo_center_x, thermo_y), bulb_radius - 5)
        if fill_height > 0:
            pygame.draw.rect(self.screen, fill_color, (thermo_center_x - stem_width/2 + 5, thermo_y - (fill_height - bulb_radius/2), stem_width - 10, fill_height-bulb_radius/2), border_top_left_radius=10, border_top_right_radius=10)

        # --- Text Display ---
        self._render_and_blit_text(f"{self.current_temp:.1f}°C", self.font_big, self.COLOR['text'], (current_center_x, panel_rect.centery - 10), anchor="center")
        self._render_and_blit_text("Current", self.font_small, self.COLOR['text_light'], (current_center_x, panel_rect.centery + 25), anchor="center")
        
        self._render_and_blit_text(f"Target: {self.target_temp:.1f}°C", self.font_medium, self.COLOR['accent_red'], (status_center_x, panel_rect.centery - 60), anchor="center")

        # Only show fuzzy control action, not ambient drift
        action_text = self.control_action
        if action_text == 'COOL': bg_color, text_color = (229, 242, 255), (37, 99, 235)
        elif action_text == 'HEAT': bg_color, text_color = (254, 226, 226), (220, 38, 38)
        else: bg_color, text_color = (243, 244, 246), (75, 85, 99)
        
        action_rect = pygame.Rect(0, 0, 120, 35)
        action_rect.center = (status_center_x, panel_rect.centery)
        pygame.draw.rect(self.screen, bg_color, action_rect, border_radius=15)
        self._render_and_blit_text(action_text, self.font_small, text_color, action_rect.center, anchor="center")

        # Update status text to show ambient drift if applicable
        if self.is_applying_ambient_drift:
            status, status_color = "Passive Drift", self.COLOR['accent_purple']
        else:
            is_at_target = abs(self.current_temp - self.target_temp) <= 0.4
            status, status_color = ("✓ At Target", self.COLOR['accent_green']) if is_at_target and self.is_running else ("Adjusting", self.COLOR['accent_orange'])
        
        self._render_and_blit_text(status, self.font_medium, status_color, (status_center_x, panel_rect.centery + 60), anchor="center")
    
    def draw_control_panel(self):
        """Draws the control panel with buttons and target temp display."""
        panel_rect = self.control_panel_rect
        self._draw_panel(panel_rect, "Control Panel")

        self._render_and_blit_text("Target Temperature (°C)", self.font_small, self.COLOR['text_light'], (panel_rect.x + 30, panel_rect.y + 60))
        self._render_and_blit_text(f"{self.target_temp:.1f}", self.font_medium, self.COLOR['text'], (panel_rect.x + 130, panel_rect.y + 95), anchor="center")
        
        mouse_pos = pygame.mouse.get_pos()
        
        pygame.draw.rect(self.screen, self.COLOR['grid'], self.btn_plus_rect, border_radius=8)
        self._render_and_blit_text("+", self.font_big, self.COLOR['text'], self.btn_plus_rect.center, anchor="center")
        pygame.draw.rect(self.screen, self.COLOR['grid'], self.btn_minus_rect, border_radius=8)
        self._render_and_blit_text("-", self.font_big, self.COLOR['text'], self.btn_minus_rect.center, anchor="center")
        
        sp_color = self.COLOR['btn_pause_hover'] if self.btn_start_pause_rect.collidepoint(mouse_pos) and self.is_running else self.COLOR['btn_pause'] if self.is_running else self.COLOR['btn_start_hover'] if self.btn_start_pause_rect.collidepoint(mouse_pos) else self.COLOR['btn_start']
        pygame.draw.rect(self.screen, sp_color, self.btn_start_pause_rect, border_radius=8)
        self._render_and_blit_text("Pause" if self.is_running else "Start", self.font_medium, (255,255,255), self.btn_start_pause_rect.center, anchor="center")

        reset_color = self.COLOR['btn_reset_hover'] if self.btn_reset_rect.collidepoint(mouse_pos) else self.COLOR['btn_reset']
        pygame.draw.rect(self.screen, reset_color, self.btn_reset_rect, border_radius=8)
        self._render_and_blit_text("Reset", self.font_medium, (255,255,255), self.btn_reset_rect.center, anchor="center")

    def draw_ambient_panel(self):
        """Draws the ambient mode selection panel with toggle button."""
        panel_rect = self.ambient_panel_rect
        self._draw_panel(panel_rect, "")
        pygame.draw.rect(self.screen, self.COLOR['grid'], self.btn_ambient_toggle_rect, border_radius=8)
        self._render_and_blit_text(f"Mode: {self.ambient_mode}", self.font_small, self.COLOR['text'], self.btn_ambient_toggle_rect.center, anchor="center")

    def draw_metrics_panel(self):
        """Draws the panel displaying system metrics."""
        panel_rect = self.metrics_panel_rect
        self._draw_panel(panel_rect, "System Metrics")
        
        # Add ambient drift indicator to metrics
        drift_status = "Yes" if self.is_applying_ambient_drift else "No"
        drift_color = self.COLOR['accent_purple'] if self.is_applying_ambient_drift else self.COLOR['text_light']
        
        metrics = [
            ("Error:", f"{self.error:.1f}°C", self.COLOR['accent_blue']),
            ("Error Rate:", f"{self.error_dot:.3f}°C/s", self.COLOR['accent_purple']),
            ("Action Strength:", f"{self.action_strength:.3f}", self.COLOR['accent_orange']),
            ("Runtime:", f"{self.time_step * 0.5:.1f}s", self.COLOR['accent_green'])
        ]
        
        for i, (label, value, color) in enumerate(metrics):
            item_rect = pygame.Rect(panel_rect.x + 20, panel_rect.y + 70 + i * 40, panel_rect.width - 40, 35)
            pygame.draw.rect(self.screen, self.COLOR['bg'], item_rect, border_radius=8)
            self._render_and_blit_text(label, self.font_small, self.COLOR['text'], (item_rect.left + 15, item_rect.centery), anchor="midleft")
            self._render_and_blit_text(value, self.font_mono, color, (item_rect.right - 15, item_rect.centery), anchor="midright")

    def process_events(self):
        """Handles all user input events."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False 
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if self.btn_start_pause_rect.collidepoint(event.pos):
                    self.is_running = not self.is_running
                    if self.is_running:
                       self.last_update_time = pygame.time.get_ticks()
                       self.prev_error = self.target_temp - self.current_temp 
                elif self.btn_reset_rect.collidepoint(event.pos):
                    self.reset_simulation()
                elif self.btn_plus_rect.collidepoint(event.pos):
                    self.target_temp = min(100.0, round(self.target_temp + 0.5, 1))
                elif self.btn_minus_rect.collidepoint(event.pos):
                    self.target_temp = max(0.0, round(self.target_temp - 0.5, 1))
                elif self.btn_ambient_toggle_rect.collidepoint(event.pos):
                    if self.ambient_mode == 'Summer':
                        self.ambient_mode, self.ambient_temp = 'Winter', 10.0
                    else:
                        self.ambient_mode, self.ambient_temp = 'Summer', 30.0
        return True

    def run(self):
        """Main application loop."""
        running = True
        while running:
            running = self.process_events()

            if self.is_running:
                current_time = pygame.time.get_ticks()
                if current_time - self.last_update_time >= 500:
                    self.update_simulation()
                    self.last_update_time = current_time

            self.screen.fill(self.COLOR['bg'])
            self._render_and_blit_text("Fuzzy Logic Thermal Control System", self.font_big, self.COLOR['text'], (self.WIDTH / 2, 40), anchor="center")
            
            self.draw_graph()
            self.draw_thermometer_display()
            self.draw_control_panel()
            self.draw_ambient_panel()
            self.draw_metrics_panel()

            pygame.display.flip()
            self.clock.tick(60)

        pygame.quit()

# --- Run the application ---
if __name__ == '__main__':
    app = FuzzyThermalControl()
    app.run()