import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import skfuzzy as fuzz
from skfuzzy import control as ctrl
from datetime import datetime
import time
import threading

# Page configuration
st.set_page_config(
    page_title="Fuzzy Logic Air Conditioning System",
    page_icon="🌡️",
    layout="wide"
)

# Initialize session state
if 'history' not in st.session_state:
    st.session_state.history = []
if 'simulation_running' not in st.session_state:
    st.session_state.simulation_running = False
if 'simulation_data' not in st.session_state:
    st.session_state.simulation_data = []
if 'current_sim_temp' not in st.session_state:
    st.session_state.current_sim_temp = 75.0

def create_fuzzy_system():
    """Create the fuzzy logic control system"""
    # Define input and output variables
    error = ctrl.Antecedent(np.arange(-10, 11, 0.1), 'error')
    error_dot = ctrl.Antecedent(np.arange(-15, 16, 0.1), 'error_dot')
    cooling = ctrl.Consequent(np.arange(0, 1.01, 0.01), 'cooling')
    
    # Define membership functions for error
    error['negative'] = fuzz.trapmf(error.universe, [-10, -10, -4, 0])
    error['zero'] = fuzz.trimf(error.universe, [-2, 0, 2])
    error['positive'] = fuzz.trapmf(error.universe, [0, 4, 10, 10])
    
    # Define membership functions for error_dot
    error_dot['negative'] = fuzz.trapmf(error_dot.universe, [-15, -15, -10, 0])
    error_dot['zero'] = fuzz.trimf(error_dot.universe, [-5, 0, 5])
    error_dot['positive'] = fuzz.trapmf(error_dot.universe, [0, 10, 15, 15])
    
    # Define membership functions for cooling output
    cooling['off'] = fuzz.trapmf(cooling.universe, [0, 0, 0.3, 0.5])
    cooling['on'] = fuzz.trapmf(cooling.universe, [0.5, 0.7, 1, 1])
    
    # Define fuzzy rules
    rule1 = ctrl.Rule(error['negative'], cooling['on'])  # Too hot -> cooling on
    rule2 = ctrl.Rule(error['zero'], cooling['off'])     # Perfect -> cooling off
    rule3 = ctrl.Rule(error['positive'], cooling['off']) # Too cold -> cooling off
    rule4 = ctrl.Rule(error_dot['positive'], cooling['on'])  # Getting hotter -> cooling on
    rule5 = ctrl.Rule(error_dot['negative'], cooling['off']) # Getting colder -> cooling off
    
    # Create control system
    cooling_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5])
    cooling_sim = ctrl.ControlSystemSimulation(cooling_ctrl)
    
    return cooling_sim, error, error_dot, cooling

def simulate_temperature_response(current_temp, target_temp, cooling_output, dt=0.1, room_thermal_mass=0.1):
    """
    Simulate how the room temperature changes based on cooling output
    Args:
        current_temp: Current room temperature
        target_temp: Target temperature
        cooling_output: Fuzzy system cooling output (0-1)
        dt: Time step
        room_thermal_mass: Thermal inertia of the room (lower = faster response)
    """
    # Environmental heat gain (room naturally warms up)
    ambient_temp = 80.0  # Ambient temperature
    heat_gain = (ambient_temp - current_temp) * 0.02  # Natural heat gain
    
    # Cooling effect based on fuzzy output
    cooling_effect = -cooling_output * 5.0  # Max cooling rate of 5°F per time step
    
    # Apply thermal mass (slows down temperature changes)
    temp_change = (heat_gain + cooling_effect) * room_thermal_mass * dt
    
    # Update temperature with some smoothing to prevent oscillations
    new_temp = current_temp + temp_change
    
    return new_temp

def calculate_error_dot(current_error, history=None):
    """Calculate error dot (rate of change)"""
    if history is None:
        history = st.session_state.history
        
    if len(history) == 0:
        return 0.0
    
    previous_error = history[-1]['error']
    return previous_error - current_error

def run_simulation_step(target_temp, cooling_sim, thermal_mass, dt):
    """Run one step of the simulation"""
    current_temp = st.session_state.current_sim_temp
    
    # Calculate error and error_dot
    current_error = target_temp - current_temp
    
    # Use simulation data for error_dot if available
    if len(st.session_state.simulation_data) > 0:
        prev_error = st.session_state.simulation_data[-1]['error']
        current_error_dot = (prev_error - current_error) / dt
    else:
        current_error_dot = 0.0
    
    # Get fuzzy system output
    try:
        cooling_sim.input['error'] = current_error
        cooling_sim.input['error_dot'] = current_error_dot
        cooling_sim.compute()
        cooling_output = cooling_sim.output['cooling']
    except:
        cooling_output = 0.0
    
    # Simulate temperature response
    new_temp = simulate_temperature_response(
        current_temp, target_temp, cooling_output, dt, thermal_mass
    )
    
    # Update current temperature
    st.session_state.current_sim_temp = new_temp
    
    # Store simulation data
    sim_data = {
        'time': len(st.session_state.simulation_data) * dt,
        'temperature': new_temp,
        'target': target_temp,
        'error': current_error,
        'error_dot': current_error_dot,
        'cooling_output': cooling_output,
        'cooling_status': 'ON' if cooling_output > 0.5 else 'OFF'
    }
    
    st.session_state.simulation_data.append(sim_data)
    
    return sim_data

def plot_membership_functions(fuzzy_var, current_value, title):
    """Plot membership functions for a fuzzy variable"""
    fig = go.Figure()
    
    colors = {'negative': '#e74c3c', 'zero': '#27ae60', 'positive': '#3498db',
              'off': '#3498db', 'on': '#e74c3c'}
    
    for label in fuzzy_var.terms:
        mf = fuzzy_var[label].mf
        fig.add_trace(go.Scatter(
            x=fuzzy_var.universe,
            y=mf,
            mode='lines',
            name=label.title(),
            line=dict(color=colors.get(label, '#2c3e50'), width=3)
        ))
    
    # Add current value line
    fig.add_vline(
        x=current_value,
        line_dash="dash",
        line_color="#f39c12",
        line_width=4,
        annotation_text="Current Value"
    )
    
    fig.update_layout(
        title=title,
        xaxis_title=fuzzy_var.label.title(),
        yaxis_title="Membership Degree",
        yaxis=dict(range=[0, 1.1]),
        height=400,
        template="plotly_white"
    )
    
    return fig

def plot_output(cooling_var, cooling_output):
    """Plot the fuzzy output and defuzzification"""
    fig = go.Figure()
    
    # Plot membership functions
    for label in cooling_var.terms:
        mf = cooling_var[label].mf
        fig.add_trace(go.Scatter(
            x=cooling_var.universe,
            y=mf,
            mode='lines',
            name=f'Cooling {label.upper()}',
            line=dict(width=2)
        ))
    
    # Add defuzzified output line
    fig.add_vline(
        x=cooling_output,
        line_dash="dash",
        line_color="#8e44ad",
        line_width=4,
        annotation_text=f"Output: {cooling_output:.3f}"
    )
    
    fig.update_layout(
        title="Fuzzy Output and Defuzzification",
        xaxis_title="Cooling Output",
        yaxis_title="Membership Degree",
        yaxis=dict(range=[0, 1.1]),
        height=400,
        template="plotly_white"
    )
    
    return fig

def plot_simulation_results():
    """Plot the simulation results"""
    if not st.session_state.simulation_data:
        return None
    
    df = pd.DataFrame(st.session_state.simulation_data)
    
    # Create subplots
    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=('Temperature Response', 'Temperature Error', 'Cooling Output'),
        vertical_spacing=0.08,
        shared_xaxes=True
    )
    
    # Temperature plot
    fig.add_trace(
        go.Scatter(
            x=df['time'],
            y=df['temperature'],
            name='Room Temperature',
            line=dict(color='red', width=3),
            mode='lines'
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=df['time'],
            y=df['target'],
            name='Target Temperature',
            line=dict(color='green', width=2, dash='dash'),
            mode='lines'
        ),
        row=1, col=1
    )
    
    # Error plot
    fig.add_trace(
        go.Scatter(
            x=df['time'],
            y=df['error'],
            name='Temperature Error',
            line=dict(color='blue', width=2),
            mode='lines',
            showlegend=False
        ),
        row=2, col=1
    )
    
    # Add zero line for error
    fig.add_hline(y=0, line_dash="dot", line_color="gray", row=2, col=1)
    
    # Cooling output plot
    fig.add_trace(
        go.Scatter(
            x=df['time'],
            y=df['cooling_output'],
            name='Cooling Output',
            line=dict(color='purple', width=2),
            mode='lines',
            showlegend=False
        ),
        row=3, col=1
    )
    
    # Add threshold line for cooling
    fig.add_hline(y=0.5, line_dash="dot", line_color="gray", row=3, col=1)
    
    fig.update_layout(
        height=800,
        template="plotly_white",
        title="Fuzzy Logic AC System Simulation"
    )
    
    fig.update_xaxes(title_text="Time (minutes)", row=3, col=1)
    fig.update_yaxes(title_text="Temperature (°F)", row=1, col=1)
    fig.update_yaxes(title_text="Error (°F)", row=2, col=1)
    fig.update_yaxes(title_text="Cooling Output", row=3, col=1, range=[0, 1.1])
    
    return fig

def main():
    st.title("🌡️ Fuzzy Logic Air Conditioning System with Simulation")
    st.markdown("---")
    
    # Create fuzzy system
    cooling_sim, error_var, error_dot_var, cooling_var = create_fuzzy_system()
    
    # Sidebar controls
    st.sidebar.header("🎛️ Control Panel")
    
    # Simulation controls
    st.sidebar.subheader("Simulation Controls")
    
    target_temp = st.sidebar.slider(
        "Target Temperature (°F)",
        min_value=-100.0,
        max_value=85.0,
        value=72.0,
        step=0.5
    )
    
    initial_temp = st.sidebar.slider(
        "Initial Room Temperature (°F)",
        min_value=-100.0,
        max_value=85.0,
        value=75.0,
        step=0.5
    )
    
    # Simulation parameters
    st.sidebar.subheader("Simulation Parameters")
    
    thermal_mass = st.sidebar.slider(
        "Room Thermal Mass (Response Rate)",
        min_value=0.01,
        max_value=0.5,
        value=0.1,
        step=0.01,
        help="Lower values = faster response, Higher values = slower response (prevents ripples)"
    )
    
    sim_speed = st.sidebar.slider(
        "Simulation Speed",
        min_value=0.1,
        max_value=2.0,
        value=1.0,
        step=0.1,
        help="Higher values = faster simulation"
    )
    
    dt = 0.1 / sim_speed  # Time step
    
    # Simulation control buttons
    col1, col2 = st.sidebar.columns(2)
    
    with col1:
        if st.button("▶️ Start Simulation", type="primary"):
            st.session_state.simulation_running = True
            st.session_state.simulation_data = []
            st.session_state.current_sim_temp = initial_temp
    
    with col2:
        if st.button("⏹️ Stop Simulation"):
            st.session_state.simulation_running = False
    
    if st.sidebar.button("🔄 Reset Simulation"):
        st.session_state.simulation_running = False
        st.session_state.simulation_data = []
        st.session_state.current_sim_temp = initial_temp
        st.rerun()
    
    # Manual controls
    st.sidebar.subheader("Manual Controls")
    
    manual_target = st.sidebar.slider(
        "Manual Target Temperature (°F)",
        min_value=60.0,
        max_value=85.0,
        value=72.0,
        step=0.5,
        key="manual_target"
    )
    
    manual_room = st.sidebar.slider(
        "Manual Room Temperature (°F)",
        min_value=60.0,
        max_value=85.0,
        value=75.0,
        step=0.5,
        key="manual_room"
    )
    
    if st.sidebar.button("Add Manual Reading"):
        error = manual_target - manual_room
        error_dot = calculate_error_dot(error)
        
        reading = {
            'timestamp': datetime.now(),
            'target_temp': manual_target,
            'room_temp': manual_room,
            'error': error,
            'error_dot': error_dot
        }
        
        st.session_state.history.append(reading)
        st.sidebar.success("Manual reading added!")
    
    # Main content area
    # Create placeholder for simulation
    sim_placeholder = st.empty()
    
    # Run simulation if active
    if st.session_state.simulation_running:
        with sim_placeholder.container():
            # Current simulation status
            if st.session_state.simulation_data:
                latest_data = st.session_state.simulation_data[-1]
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Current Temperature", f"{latest_data['temperature']:.1f}°F")
                
                with col2:
                    st.metric("Target Temperature", f"{latest_data['target']:.1f}°F")
                
                with col3:
                    st.metric("Temperature Error", f"{latest_data['error']:.1f}°F")
                
                with col4:
                    cooling_status = latest_data['cooling_status']
                    st.metric("Cooling Status", cooling_status, f"{latest_data['cooling_output']:.1%}")
                
                # Real-time plot
                fig = plot_simulation_results()
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
                
                # Progress indicator
                if abs(latest_data['error']) < 0.5:
                    st.success("🎯 Target temperature reached!")
                elif abs(latest_data['error']) < 2.0:
                    st.warning("🔄 Approaching target temperature...")
                else:
                    st.info("🌡️ System is actively working to reach target temperature...")
            
            # Run simulation step
            if len(st.session_state.simulation_data) < 500:  # Limit simulation length
                sim_data = run_simulation_step(target_temp, cooling_sim, thermal_mass, dt)
                time.sleep(0.1)  # Small delay for animation effect
                st.rerun()
            else:
                st.session_state.simulation_running = False
                st.success("Simulation completed!")
    
    else:
        # Static analysis mode
        current_error = manual_target - manual_room
        current_error_dot = calculate_error_dot(current_error)
        
        # Run fuzzy logic simulation
        try:
            cooling_sim.input['error'] = current_error
            cooling_sim.input['error_dot'] = current_error_dot
            cooling_sim.compute()
            cooling_output = cooling_sim.output['cooling']
        except:
            cooling_output = 0.0
        
        # Display current status
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Target Temperature", f"{manual_target:.1f}°F")
        
        with col2:
            st.metric("Room Temperature", f"{manual_room:.1f}°F")
        
        with col3:
            st.metric("Temperature Error", f"{current_error:.1f}°F")
        
        with col4:
            cooling_status = "ON" if cooling_output > 0.5 else "OFF"
            st.metric("Cooling Status", cooling_status, f"{cooling_output:.1%}")
        
        st.markdown("---")
        
        # Create visualization tabs
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Membership Functions", "🎛️ System Output", "📈 History", "🔬 Simulation Results"])
        
        with tab1:
            col1, col2 = st.columns(2)
            
            with col1:
                error_fig = plot_membership_functions(
                    error_var, current_error, 
                    "Temperature Error Membership Functions"
                )
                st.plotly_chart(error_fig, use_container_width=True)
            
            with col2:
                error_dot_fig = plot_membership_functions(
                    error_dot_var, current_error_dot,
                    "Error Rate Membership Functions"
                )
                st.plotly_chart(error_dot_fig, use_container_width=True)
        
        with tab2:
            output_fig = plot_output(cooling_var, cooling_output)
            st.plotly_chart(output_fig, use_container_width=True)
            
            # System interpretation
            st.subheader("System Interpretation")
            if cooling_output > 0.7:
                st.success("🔴 **Cooling System: ON** - Room is too warm, actively cooling")
            elif cooling_output < 0.3:
                st.info("🔵 **Cooling System: OFF** - Temperature is acceptable or room is too cold")
            else:
                st.warning("🟡 **Cooling System: MODERATE** - Partial cooling needed")
        
        with tab3:
            if st.session_state.history:
                # Convert history to DataFrame
                df = pd.DataFrame(st.session_state.history)
                df['time'] = df['timestamp'].dt.strftime('%H:%M:%S')
                
                # Display table
                st.subheader("Manual Temperature History")
                display_df = df[['time', 'target_temp', 'room_temp', 'error', 'error_dot']].copy()
                display_df.columns = ['Time', 'Target (°F)', 'Room (°F)', 'Error (°F)', 'Rate (°F/min)']
                st.dataframe(display_df, use_container_width=True)
                
                # Clear history button
                if st.button("Clear History"):
                    st.session_state.history = []
                    st.rerun()
            else:
                st.info("No manual temperature readings yet. Add some readings using the sidebar controls!")
        
        with tab4:
            if st.session_state.simulation_data:
                st.subheader("Latest Simulation Results")
                fig = plot_simulation_results()
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
                
                # Simulation statistics
                df = pd.DataFrame(st.session_state.simulation_data)
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    settling_time = None
                    for i, row in df.iterrows():
                        if abs(row['error']) < 0.5:
                            settling_time = row['time']
                            break
                    
                    if settling_time:
                        st.metric("Settling Time", f"{settling_time:.1f} min")
                    else:
                        st.metric("Settling Time", "Not reached")
                
                with col2:
                    max_overshoot = df['error'].min() if df['error'].min() < 0 else 0
                    st.metric("Max Overshoot", f"{abs(max_overshoot):.1f}°F")
                
                with col3:
                    steady_state_error = df['error'].iloc[-1] if len(df) > 0 else 0
                    st.metric("Steady State Error", f"{steady_state_error:.1f}°F")
                
            else:
                st.info("No simulation data available. Run a simulation to see results!")

if __name__ == "__main__":
    main()