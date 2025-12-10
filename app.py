"""
Streamlit UI for Palestinian Delivery TSP: SA vs GA Comparison
Palestinian Delivery App - Finding the best route to visit 50 regions with minimum cost
"""
import streamlit as st
import matplotlib.pyplot as plt
import time

# Import solvers and utilities
from tsp_sa_solver import TravelingSalesmanSA
from ga_solver import TravelingSalesmanGA
from tsp_utils import visualize_tsp_tour, calculate_tour_distance_km
from palestinian_cities import get_cities_coordinates, get_city_name, get_all_city_names

# Page configuration
st.set_page_config(page_title="Palestinian Delivery TSP App", layout="wide")

# Font support
plt.rcParams['font.family'] = 'DejaVu Sans'


def _run_sa(cities, city_names, initial_temp, min_temp, max_iterations, cooling_rate,
           show_animation, animation_speed, live_viz_placeholder, status_placeholder):
    """Run Simulated Annealing algorithm."""
    solver = TravelingSalesmanSA(cities)
    start_time = time.time()
    
    if show_animation:
        status_placeholder.info("🔄 Running Simulated Annealing with live visualization...")
        
        best_tour = None
        best_distance = float('inf')
        temp_history = []
        fitness_history = []
        
        # Create initial visualization
        initial_tour = solver.initial_config()
        fig = visualize_tsp_tour(cities, initial_tour, city_names, 
                                 "Delivery Route - Simulated Annealing", show_labels=False)
        live_viz_placeholder.pyplot(fig)
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        generator = solver.sim_anneal(max_iterations, initial_temp, yield_progress=True, 
                                     min_temp=min_temp, cooling_rate=cooling_rate)
        
        try:
            iteration = 0
            last_yield = None
            stop_reason = "max_iterations"
            
            for current_tour, iter_num, temp, distance in generator:
                iteration = iter_num
                last_yield = (current_tour, iter_num, temp, distance)
                
                temp_history.append(temp)
                fitness_history.append(distance)
                
                if temp <= min_temp:
                    stop_reason = "temperature_threshold"
                
                if distance < best_distance:
                    best_tour = current_tour.copy()
                    best_distance = distance
                
                if iteration % animation_speed == 0 or iteration == 1:
                    fig = visualize_tsp_tour(cities, current_tour, city_names,
                                           f"Delivery Route - SA | Cost: {distance:.2f} km", 
                                           show_labels=False)
                    live_viz_placeholder.pyplot(fig)
                    
                    stop_indicator = "🌡️ Temp threshold" if temp <= min_temp else ""
                    status_text.text(f"🔄 Iteration: {iteration}/{max_iterations} | "
                                   f"Distance: {distance:.2f} km | "
                                   f"Temperature: {temp:.2f} {stop_indicator}")
                    
                    progress_bar.progress(min(iteration / max_iterations, 1.0))
                    time.sleep(0.01)
            
            if hasattr(solver, '_final_result'):
                final_tour, final_dist, final_temp, final_fit, final_stop = solver._final_result
                stop_reason = final_stop
                if best_tour is None:
                    best_tour = final_tour
                    best_distance = final_dist
                if len(temp_history) == max_iterations:
                    temp_history = final_temp
                    fitness_history = final_fit
            elif hasattr(solver, '_stop_reason'):
                stop_reason = solver._stop_reason
        
        except Exception as e:
            st.error(f"Error during execution: {e}")
            import traceback
            st.code(traceback.format_exc())
            best_tour, best_distance, temp_history, fitness_history, stop_reason = solver.sim_anneal_simple(
                max_iterations, initial_temp, min_temp, cooling_rate
            )
        
        progress_bar.progress(1.0)
        
        if best_tour:
            fig = visualize_tsp_tour(cities, best_tour, city_names,
                                   f"Best Route - SA | Cost: {best_distance:.2f} km", 
                                   show_labels=False)
            live_viz_placeholder.pyplot(fig)
        
        elapsed_time = time.time() - start_time
        if stop_reason == "solution_found":
            status_placeholder.success(f"✅ Simulated Annealing completed! Optimal solution found - Time: {elapsed_time:.2f}s")
        elif stop_reason == "temperature_threshold":
            status_placeholder.success(f"✅ Simulated Annealing completed! (Stopped at temperature ≤ {min_temp}) - Time: {elapsed_time:.2f}s")
        else:
            status_placeholder.success(f"✅ Simulated Annealing completed! (Reached max iterations) - Time: {elapsed_time:.2f}s")
        status_text.empty()
        
    else:
        with st.spinner("Running Simulated Annealing..."):
            best_tour, best_distance, temp_history, fitness_history, stop_reason = solver.sim_anneal_simple(
                max_iterations, initial_temp, min_temp, cooling_rate
            )
            elapsed_time = time.time() - start_time
            if stop_reason == "solution_found":
                st.success(f"✅ Simulated Annealing completed! Optimal solution found - Time: {elapsed_time:.2f}s")
            elif stop_reason == "temperature_threshold":
                st.success(f"✅ Simulated Annealing completed! (Stopped at temperature ≤ {min_temp}) - Time: {elapsed_time:.2f}s")
            else:
                st.success(f"✅ Simulated Annealing completed! (Reached max iterations) - Time: {elapsed_time:.2f}s")
    
    st.session_state['best_tour'] = best_tour
    st.session_state['temp_history'] = temp_history
    st.session_state['fitness_history'] = fitness_history
    st.session_state['solver'] = solver
    st.session_state['cities'] = cities
    st.session_state['algorithm'] = 'SA'
    st.session_state['execution_time'] = elapsed_time
    st.session_state['stop_reason'] = stop_reason if 'stop_reason' in locals() else "max_iterations"
    st.session_state['best_distance'] = best_distance


def _run_ga(cities, city_names, population_size, generations, crossover_rate, mutation_rate,
            elitism_count, tournament_size, crossover_type, mutation_type,
            show_animation, animation_speed, live_viz_placeholder, status_placeholder):
    """Run Genetic Algorithm."""
    solver = TravelingSalesmanGA(cities)
    start_time = time.time()
    
    if show_animation:
        status_placeholder.info("🔄 Running Genetic Algorithm with live visualization...")
        
        best_tour = None
        best_distance = float('inf')
        fitness_history = []
        
        # Create initial visualization
        initial_tour = solver.create_random_tour()
        fig = visualize_tsp_tour(cities, initial_tour, city_names,
                                "Delivery Route - Genetic Algorithm", show_labels=False)
        live_viz_placeholder.pyplot(fig)
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        generator = solver.genetic_algorithm(
            population_size, generations, crossover_rate, mutation_rate,
            elitism_count, tournament_size, crossover_type, mutation_type,
            yield_progress=True
        )
        
        try:
            generation = 0
            stop_reason = "max_generations"
            
            for current_tour, gen_num, distance in generator:
                generation = gen_num
                
                fitness_history.append(distance)
                
                if distance < best_distance:
                    best_tour = current_tour.copy()
                    best_distance = distance
                
                if generation % animation_speed == 0 or generation == 1:
                    fig = visualize_tsp_tour(cities, current_tour, city_names,
                                           f"Delivery Route - GA | Cost: {distance:.2f} km | Generation: {generation}",
                                           show_labels=False)
                    live_viz_placeholder.pyplot(fig)
                    
                    status_text.text(f"🔄 Generation: {generation}/{generations} | "
                                   f"Distance: {distance:.2f} km")
                    
                    progress_bar.progress(min(generation / generations, 1.0))
                    time.sleep(0.01)
            
            if hasattr(solver, '_final_result'):
                final_tour, final_dist, final_fit, final_stop = solver._final_result
                stop_reason = final_stop
                if best_tour is None:
                    best_tour = final_tour
                    best_distance = final_dist
                fitness_history = final_fit
            elif hasattr(solver, '_stop_reason'):
                stop_reason = solver._stop_reason
        
        except Exception as e:
            st.error(f"Error during execution: {e}")
            import traceback
            st.code(traceback.format_exc())
            best_tour, best_distance, fitness_history, stop_reason = solver.genetic_algorithm(
                population_size, generations, crossover_rate, mutation_rate,
                elitism_count, tournament_size, crossover_type, mutation_type,
                yield_progress=False
            )
        
        progress_bar.progress(1.0)
        
        if best_tour:
            fig = visualize_tsp_tour(cities, best_tour, city_names,
                                   f"Best Route - GA | Cost: {best_distance:.2f} km",
                                   show_labels=False)
            live_viz_placeholder.pyplot(fig)
        
        elapsed_time = time.time() - start_time
        status_placeholder.success(f"✅ Genetic Algorithm completed! - Time: {elapsed_time:.2f}s")
        status_text.empty()
        
    else:
        with st.spinner("Running Genetic Algorithm..."):
            best_tour, best_distance, fitness_history, stop_reason = solver.genetic_algorithm(
                population_size, generations, crossover_rate, mutation_rate,
                elitism_count, tournament_size, crossover_type, mutation_type,
                yield_progress=False
            )
            elapsed_time = time.time() - start_time
            st.success(f"✅ Genetic Algorithm completed! - Time: {elapsed_time:.2f}s")
    
    st.session_state['best_tour'] = best_tour
    st.session_state['fitness_history'] = fitness_history
    st.session_state['solver'] = solver
    st.session_state['cities'] = cities
    st.session_state['algorithm'] = 'GA'
    st.session_state['execution_time'] = elapsed_time
    st.session_state['stop_reason'] = stop_reason if 'stop_reason' in locals() else "max_generations"
    st.session_state['best_distance'] = best_distance


def _compare_algorithms(cities, city_names, initial_temp, min_temp, max_iterations, cooling_rate,
                       population_size, generations, crossover_rate, mutation_rate,
                       elitism_count, tournament_size, crossover_type, mutation_type,
                       show_animation, animation_speed, live_viz_placeholder, status_placeholder):
    """Compare both SA and GA algorithms."""
    status_placeholder.info("🔄 Running both algorithms for comparison...")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Simulated Annealing")
        try:
            sa_start = time.time()
            sa_solver = TravelingSalesmanSA(cities)
            result = sa_solver.sim_anneal_simple(max_iterations, initial_temp, min_temp, cooling_rate)
            sa_tour, sa_distance, sa_temp, sa_fit, sa_stop = result
            sa_time = time.time() - sa_start
            
            st.metric("Execution Time", f"{sa_time:.3f}s")
            st.metric("Total Distance", f"{sa_distance:.2f} km")
            st.metric("Iterations", max_iterations)
            
        except Exception as e:
            st.error(f"Error in SA: {e}")
            sa_tour = None
            sa_time = 0
            sa_distance = float('inf')
            sa_stop = "error"
    
    with col2:
        st.subheader("Genetic Algorithm")
        try:
            ga_start = time.time()
            ga_solver = TravelingSalesmanGA(cities)
            
            # Call _genetic_algorithm_core directly to avoid generator issue
            result = ga_solver._genetic_algorithm_core(
                population_size, generations, crossover_rate, mutation_rate,
                elitism_count, tournament_size, crossover_type, mutation_type
            )
            
            # Validate result
            if result is None:
                raise ValueError("_genetic_algorithm_core returned None")
            if not isinstance(result, tuple):
                raise ValueError(f"_genetic_algorithm_core returned {type(result)}, expected tuple")
            if len(result) != 4:
                raise ValueError(f"Expected 4 values from _genetic_algorithm_core, got {len(result)}")
            
            ga_tour, ga_distance, ga_fit, ga_stop = result
            ga_time = time.time() - ga_start
            
            st.metric("Execution Time", f"{ga_time:.3f}s")
            st.metric("Total Distance", f"{ga_distance:.2f} km")
            st.metric("Generations", generations)
            
        except Exception as e:
            st.error(f"Error in GA: {e}")
            import traceback
            st.code(traceback.format_exc())
            ga_tour = None
            ga_time = 0
            ga_distance = float('inf')
            ga_stop = "error"
    
    # Comparison summary
    st.markdown("---")
    st.subheader("📊 Comparison Summary")
    
    comp_col1, comp_col2, comp_col3 = st.columns(3)
    
    with comp_col1:
        if sa_tour and ga_tour:
            if sa_distance < ga_distance:
                improvement = ((ga_distance - sa_distance) / sa_distance) * 100
                st.success(f"🏆 SA found better route by {ga_distance - sa_distance:.2f} km ({improvement:.1f}% better)")
                st.caption("💡 Try increasing GA population size (100-200) and generations (200-500) for better results")
            elif ga_distance < sa_distance:
                improvement = ((sa_distance - ga_distance) / ga_distance) * 100
                st.success(f"🏆 GA found better route by {sa_distance - ga_distance:.2f} km ({improvement:.1f}% better)")
                st.caption("✅ GA's population-based approach found a superior solution!")
            else:
                st.info("🤝 Both algorithms found same quality solution")
        else:
            st.warning("⚠️ An error occurred in one of the algorithms")
    
    with comp_col2:
        if sa_time > 0 and ga_time > 0:
            if sa_time < ga_time:
                st.info(f"⚡ SA was {ga_time/sa_time:.2f}x faster")
                st.caption("SA is typically faster due to simpler single-solution approach")
            else:
                st.info(f"⚡ GA was {sa_time/ga_time:.2f}x faster")
    
    with comp_col3:
        # Calculate improvement from random tour
        random_tour = list(cities.keys())
        random_distance = calculate_tour_distance_km(cities, random_tour)
        improvement_sa = ((random_distance - sa_distance) / random_distance) * 100 if sa_tour else 0
        improvement_ga = ((random_distance - ga_distance) / random_distance) * 100 if ga_tour else 0
        st.info(f"📈 vs Random: SA {improvement_sa:.1f}% | GA {improvement_ga:.1f}% better")
    
    # Store results
    st.session_state['sa_tour'] = sa_tour
    st.session_state['ga_tour'] = ga_tour
    st.session_state['sa_solver'] = sa_solver if 'sa_solver' in locals() else None
    st.session_state['ga_solver'] = ga_solver if 'ga_solver' in locals() else None
    st.session_state['cities'] = cities
    st.session_state['algorithm'] = 'COMPARE'
    st.session_state['sa_time'] = sa_time if 'sa_time' in locals() else 0
    st.session_state['ga_time'] = ga_time if 'ga_time' in locals() else 0
    st.session_state['sa_distance'] = sa_distance if 'sa_distance' in locals() else 0
    st.session_state['ga_distance'] = ga_distance if 'ga_distance' in locals() else 0
    
    status_placeholder.success("✅ Comparison completed!")


def main():
    st.title("🚚 Palestinian Delivery TSP Application")
    st.markdown("### Finding the best route to visit selected Palestinian regions with minimum cost")
    
    # Add info about algorithms
    with st.expander("ℹ️ When to use SA vs GA?"):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            **Simulated Annealing (SA):**
            - ✅ Faster execution
            - ✅ Good for small-medium problems
            - ✅ Less memory usage
            - ⚠️ May get stuck in local optima
            - ⚠️ Single solution path
            """)
        with col2:
            st.markdown("""
            **Genetic Algorithm (GA):**
            - ✅ Better exploration of solution space
            - ✅ Multiple solutions evolve together
            - ✅ Better for complex problems
            - ⚠️ Slower (needs more time)
            - ⚠️ Needs larger population/generations
            """)
        st.markdown("""
        **💡 Recommendation:** 
        - Use **SA** for quick results or small problems (< 20 cities)
        - Use **GA** for better quality solutions or larger problems (> 20 cities)
        - **GA needs proper tuning**: Try population_size=100-200, generations=200-500 for best results
        - **Why GA is used**: GA explores multiple solutions simultaneously and can escape local optima better than SA
        """)
    
    st.markdown("---")
    
    # Load all Palestinian cities
    all_cities = get_cities_coordinates()
    all_city_names = get_all_city_names()
    
    # Sidebar for configuration
    st.sidebar.header("⚙️ Configuration")
    
    # City selection
    st.sidebar.subheader("0. Select Cities")
    use_all_cities = st.sidebar.checkbox("Use All 50 Cities", value=True)
    
    if use_all_cities:
        cities = all_cities
        city_names = all_city_names
    else:
        st.sidebar.markdown("**Select cities to visit:**")
        selected_city_ids = st.sidebar.multiselect(
            "Choose Cities",
            options=list(all_city_names.keys()),
            format_func=lambda x: f"{x}: {all_city_names[x]}",
            default=list(all_city_names.keys())[:10]  # Default to first 10
        )
        
        if len(selected_city_ids) < 2:
            st.sidebar.warning("⚠️ Please select at least 2 cities!")
            cities = {}
            city_names = {}
        else:
            cities = {city_id: all_cities[city_id] for city_id in selected_city_ids}
            city_names = {city_id: all_city_names[city_id] for city_id in selected_city_ids}
    
    # Algorithm selection
    st.sidebar.subheader("1. Algorithm Selection")
    algorithm_mode = st.sidebar.radio(
        "Choose Algorithm",
        ["Simulated Annealing (SA)", "Genetic Algorithm (GA)", "Compare Both"]
    )
    
    # SA Parameters
    st.sidebar.subheader("2. Simulated Annealing Parameters")
    st.sidebar.markdown("*💡 Tip: SA is faster but may get stuck in local optima*")
    initial_temp = st.sidebar.slider("Initial Temperature", min_value=100.0, max_value=10000.0, value=5000.0, step=100.0,
                                     help="Higher = more exploration initially")
    min_temp = st.sidebar.slider("Minimum Temperature", min_value=0.1, max_value=10.0, value=2.0, step=0.1)
    cooling_rate = st.sidebar.slider("Cooling Rate", min_value=0.85, max_value=0.99, value=0.95, step=0.01,
                                    help="Lower = cools faster, may miss better solutions")
    max_iterations = st.sidebar.slider("Max Iterations", min_value=100, max_value=5000, value=1000, step=100)
    
    # GA Parameters
    st.sidebar.subheader("3. Genetic Algorithm Parameters")
    st.sidebar.markdown("*💡 Tip: GA works better with larger population and more generations*")
    population_size = st.sidebar.slider("Population Size", min_value=20, max_value=300, value=100, step=10,
                                        help="Larger population = better exploration but slower. Recommended: 100-200")
    generations = st.sidebar.slider("Number of Generations", min_value=50, max_value=1000, value=200, step=10,
                                    help="More generations = better results but slower. Recommended: 200-500")
    crossover_rate = st.sidebar.slider("Crossover Rate", min_value=0.5, max_value=1.0, value=0.8, step=0.05)
    mutation_rate = st.sidebar.slider("Mutation Rate", min_value=0.01, max_value=0.5, value=0.1, step=0.01)
    elitism_count = st.sidebar.slider("Elitism Count", min_value=1, max_value=10, value=2, step=1)
    tournament_size = st.sidebar.slider("Tournament Size", min_value=2, max_value=10, value=3, step=1)
    crossover_type = st.sidebar.selectbox("Crossover Type", ["ox", "pmx"])
    mutation_type = st.sidebar.selectbox("Mutation Type", ["swap", "inversion"])
    
    # Display options
    st.sidebar.subheader("4. Display Options")
    show_animation = st.sidebar.checkbox("Show Live Animation", value=True)
    animation_speed = st.sidebar.slider("Animation Update Speed", min_value=1, max_value=50, value=5, 
                                       help="Update every N iterations/generations")
    show_progress = st.sidebar.checkbox("Show Progress Charts", value=True)
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📍 Palestinian Regions Map")
        if cities and len(cities) >= 2:
            fig = visualize_tsp_tour(cities, None, city_names, 
                                     f"Selected Regions ({len(cities)} regions)", show_labels=False)
            st.pyplot(fig)
            st.info(f"**Info:** {len(cities)} regions selected")
        else:
            st.warning("⚠️ Please select at least 2 cities to view the map")
    
    with col2:
        if algorithm_mode == "Simulated Annealing (SA)":
            st.subheader("Simulated Annealing Control")
            button_text = "🚀 Run SA"
        elif algorithm_mode == "Genetic Algorithm (GA)":
            st.subheader("Genetic Algorithm Control")
            button_text = "🚀 Run GA"
        else:
            st.subheader("Algorithm Comparison")
            button_text = "🚀 Run Comparison"
        
        live_viz_placeholder = st.empty()
        status_placeholder = st.empty()
        
        if st.button(button_text, type="primary"):
            if not cities or len(cities) < 2:
                status_placeholder.error("❌ Please select at least 2 cities to run the algorithm!")
            else:
                if algorithm_mode == "Simulated Annealing (SA)":
                    _run_sa(cities, city_names, initial_temp, min_temp, max_iterations, cooling_rate,
                           show_animation, animation_speed, live_viz_placeholder, status_placeholder)
                elif algorithm_mode == "Genetic Algorithm (GA)":
                    _run_ga(cities, city_names, population_size, generations, crossover_rate, mutation_rate,
                           elitism_count, tournament_size, crossover_type, mutation_type,
                           show_animation, animation_speed, live_viz_placeholder, status_placeholder)
                else:
                    _compare_algorithms(cities, city_names, initial_temp, min_temp, max_iterations, cooling_rate,
                                     population_size, generations, crossover_rate, mutation_rate,
                                     elitism_count, tournament_size, crossover_type, mutation_type,
                                     show_animation, animation_speed, live_viz_placeholder, status_placeholder)
    
    # Results section
    algorithm = st.session_state.get('algorithm', 'SA')
    
    if algorithm == 'COMPARE' and 'sa_tour' in st.session_state:
        st.markdown("---")
        st.subheader("📊 Detailed Comparison Results")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Simulated Annealing Result")
            sa_tour = st.session_state['sa_tour']
            sa_solver = st.session_state['sa_solver']
            cities = st.session_state['cities']
            
            if sa_tour:
                sa_distance = st.session_state.get('sa_distance', 0)
                st.metric("Total Distance", f"{sa_distance:.2f} km")
                st.metric("Execution Time", f"{st.session_state.get('sa_time', 0):.3f}s")
                
                fig_sa = visualize_tsp_tour(cities, sa_tour, city_names,
                                          f"SA Route - {sa_distance:.2f} km", show_labels=False)
                st.pyplot(fig_sa)
        
        with col2:
            st.subheader("Genetic Algorithm Result")
            ga_tour = st.session_state.get('ga_tour')
            ga_solver = st.session_state.get('ga_solver')
            
            if ga_tour:
                ga_distance = st.session_state.get('ga_distance', 0)
                st.metric("Total Distance", f"{ga_distance:.2f} km")
                st.metric("Execution Time", f"{st.session_state.get('ga_time', 0):.3f}s")
                
                fig_ga = visualize_tsp_tour(cities, ga_tour, city_names,
                                          f"GA Route - {ga_distance:.2f} km", show_labels=False)
                st.pyplot(fig_ga)
    
    elif 'best_tour' in st.session_state and st.session_state['best_tour']:
        st.markdown("---")
        st.subheader("📊 Results")
        
        best_tour = st.session_state['best_tour']
        solver = st.session_state['solver']
        cities = st.session_state['cities']
        best_distance = st.session_state.get('best_distance', 0)
        
        # Display results in columns
        if algorithm == 'SA':
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Distance", f"{best_distance:.2f} km")
            
            with col2:
                st.metric("Number of Regions", len(best_tour))
            
            with col3:
                st.metric("Execution Time", f"{st.session_state.get('execution_time', 0):.2f}s")
            
            with col4:
                stop_reason = st.session_state.get('stop_reason', 'max_iterations')
                if stop_reason == "solution_found":
                    st.metric("Stop Reason", "✅ Optimal Solution")
                elif stop_reason == "temperature_threshold":
                    st.metric("Stop Reason", "🌡️ Temperature")
                else:
                    st.metric("Stop Reason", "🔄 Max Iterations")
        
        else:  # GA
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Distance", f"{best_distance:.2f} km")
            
            with col2:
                st.metric("Number of Regions", len(best_tour))
            
            with col3:
                st.metric("Execution Time", f"{st.session_state.get('execution_time', 0):.2f}s")
            
            with col4:
                stop_reason = st.session_state.get('stop_reason', 'max_generations')
                st.metric("Stop Reason", "🔄 Max Generations")
        
        # Tour details table
        if best_tour:
            st.subheader("Tour Details")
            tour_data = []
            total_dist = 0
            for i, city_id in enumerate(best_tour):
                next_city_id = best_tour[(i + 1) % len(best_tour)]
                segment_dist = calculate_tour_distance_km(cities, [city_id, next_city_id])
                total_dist += segment_dist
                tour_data.append({
                    "Order": i + 1,
                    "Region": city_names.get(city_id, f"City {city_id}"),
                    "Cumulative Distance": f"{total_dist:.2f} km"
                })
            
            st.dataframe(tour_data, use_container_width=True)
            
            # Visualize tour
            st.subheader("Final Delivery Route")
            fig_colored = visualize_tsp_tour(cities, best_tour, city_names,
                                            f"Best Route - {best_distance:.2f} km", show_labels=False)
            st.pyplot(fig_colored)
        
        # Progress visualization
        if algorithm == 'SA' and show_progress and 'temp_history' in st.session_state:
            st.subheader("Algorithm Progress")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Temperature Over Time**")
                fig_temp = plt.figure(figsize=(10, 4))
                plt.plot(st.session_state['temp_history'])
                plt.xlabel("Iteration")
                plt.ylabel("Temperature")
                plt.title("Temperature Cooling Schedule")
                plt.grid(True, alpha=0.3)
                st.pyplot(fig_temp)
            
            with col2:
                st.markdown("**Distance Over Time**")
                fig_fitness = plt.figure(figsize=(10, 4))
                fitness_history = st.session_state.get('fitness_history', [])
                if fitness_history:
                    plt.plot(fitness_history)
                    plt.xlabel("Iteration")
                    plt.ylabel("Distance (km)")
                    plt.title("Solution Quality Improvement")
                    plt.grid(True, alpha=0.3)
                    st.pyplot(fig_fitness)
                    
                    best_iteration = min(range(len(fitness_history)), key=lambda i: fitness_history[i])
                    st.info(f"✨ Best solution found at iteration {best_iteration + 1} with distance {fitness_history[best_iteration]:.2f} km")
        
        elif algorithm == 'GA' and show_progress and 'fitness_history' in st.session_state:
            st.subheader("Algorithm Progress")
            
            st.markdown("**Distance Across Generations**")
            fig_fitness = plt.figure(figsize=(12, 5))
            fitness_history = st.session_state.get('fitness_history', [])
            if fitness_history:
                plt.plot(fitness_history)
                plt.xlabel("Generation")
                plt.ylabel("Distance (km)")
                plt.title("Solution Quality Improvement Across Generations")
                plt.grid(True, alpha=0.3)
                st.pyplot(fig_fitness)
                
                best_generation = min(range(len(fitness_history)), key=lambda i: fitness_history[i])
                st.info(f"✨ Best solution found at generation {best_generation + 1} with distance {fitness_history[best_generation]:.2f} km")


if __name__ == "__main__":
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        if get_script_run_ctx() is None:
            raise RuntimeError("Not running in Streamlit")
        main()
    except (ImportError, RuntimeError):
        import sys
        print("\n" + "="*60)
        print("ERROR: This is a Streamlit application!")
        print("="*60)
        print("\nTo run this app, use the following command:")
        print(f"\n    streamlit run app.py\n")
        print("="*60 + "\n")
        sys.exit(1)
