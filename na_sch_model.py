import numpy as np
import matplotlib.pyplot as plt
import os

from numba import jit

# Plot Style, einfach ignorieren
plt.style.use(os.path.join(os.path.dirname(os.path.realpath(__file__)), "neat.mplstyle"))
plt.rcParams['text.latex.preamble'] = r"\usepackage{lmodern}"
# Options
params = {'font.family': 'lmodern'}
plt.rcParams.update(params)

# function that creates a random state with a specific density

def create_density(rho, L):

    state = np.zeros(L)

    for i in range(int(rho*L)):

        state[i] = 1

    np.random.shuffle(state)

    return state

# function implementing the acceleration phase, every velocity is increased

def acceleration_phase(state, velocity, v_max):

    velocity = np.where(state > 0, np.minimum(velocity+1, v_max), 0)

    return state, velocity

def test_acceleration_phase():

    state = create_density(0.5, 10)
    veloc = np.arange(0,10,1)

    state, velocities = acceleration_phase(state, veloc, 5)

    print(velocities)

# implementation of the deceleration phase

def deceleration_phase(state, old_velocity):

    L = len(state)

    # calculate headway

    headway = calculate_headway(state)
    
    # get active agents

    active_agents = np.argwhere(state>0).flatten()

    velocity = np.zeros(L)

    for agent in active_agents:
        
        # for each agent, adjust velocity

        velocity[agent] = np.minimum(headway[agent], old_velocity[agent])

    return state, velocity

def test_deceleration_phase():

    state, velocities = deceleration_phase(np.array([0,1,0,0,1]), np.array([0,5,0,0,1]))

    print(velocities)

    state, velocities = deceleration_phase(np.array([0,1,0,0,0,1,0]), np.array([0,5,0,0,0,3,0]))

    print(velocities)

# function that calculates the headway

@jit(nopython=True)

def calculate_headway(state):

    L = len(state)
    
    # we first copy the state two times, periodic boundary conditions ,,,

    state = np.repeat(state, 2).reshape((-1, 2)).T.flatten() # numba compatible version of np.tile(state, 2), I'm sorry :(

    array_headway = np.zeros(L)

    current_site = 0
    current_headway = 0

    headway = False
    
    # with this loop we basically count the number of zeros between two non-zero values, implementing periodic boundary conditions

    for i in range(2*L):

        if current_site >= L:

            break
         
         # has an agent and calculate the headway for that

        if headway:
              
            if state[i] != 0:
                   
                array_headway[current_site] = current_headway

                current_site = 0
                current_headway = 0

                headway = False

            else:

                current_headway = current_headway + 1 
              
        # look for an agent to calc headway for

        if state[i] != 0 and not headway:

            current_site = i
            headway = True

    return array_headway

def test_headway():
    
    print(calculate_headway([0,0,0,1,1,0,0,1,0,0,1,0]))

# implementation of the randomization phase

@jit(nopython=True)

def randomization_phase(state, velocity, p):
    
    # p is an array of probabilities, important for the VDR model

    for i in range(len(velocity)):

        veloc = velocity[i]

        if veloc > 0:

            prob = np.random.random_sample()
            
            # with a given probability, decrease velocity

            if prob < p[i]:

                velocity[i] = veloc - 1
    
    return state, velocity

# movement phase, each car gets moved by its velocity

@jit(nopython=True)

def movement_phase(state, velocity):

    new_state = np.copy(state)
    new_velocity = np.copy(velocity)

    L = len(state)

    for i in range(L):

        if state[i] != 0 and velocity[i] != 0:


            new_index = i + velocity[i]
            
            # implementing periodic boundary conditions

            if new_index >= L:

                new_index = new_index%L

            new_index = int(new_index)

            if(state[new_index]!=0):

                print("error: site ", new_index, " is already taken")
                
            # movement of the cars + velocity, parallel update

            new_state[new_index] = 1
            new_state[i] = 0

            new_velocity[new_index] = velocity[i]
            new_velocity[i] = 0

    return new_state, new_velocity

# calculates the probability array

def calculate_prob(velocity, p0, p):

    return np.where(velocity > 0, p, p0)

# calculates the density

def calc_density(states):

    return np.mean(states, axis = 0)

# calculates the velocity

def calc_velocity(velocities, states):

    M = len(states)

    mean_veloc = []

    for i in range(M):

        state = states[i]
        velocity = velocities[i]
        
        N = np.sum(state)

        mean_veloc.append((1/N)*np.sum(velocity))

    return np.array(mean_veloc)

# calculates the flux

def calc_flux(velocities):

    return np.mean(velocities, axis = 0)


#implementation of the Na-Sch model

def na_sch_model(rho, L, p0, p, v_max, steps, VDR = False):
    
    #create starting state

    state = create_density(rho, L)
    velocity = np.zeros(L)

    states = [state]
    velocities = [velocity]
    
    # create prob array

    prob = np.ones(L)*p

    for i in range(steps):
        
        # if VDR is used, we have to calculate prob with the old velocities

        if VDR:

            # write p0 on a site, where an agent with velocity 0 sits

            prob = calculate_prob(velocity, p0, p)
            
        # do Na-Sch steps

        state, velocity = acceleration_phase(state, velocity, v_max)
        
        state, velocity = deceleration_phase(state, velocity)

        state, velocity = randomization_phase(state, velocity, prob)

        state, velocity = movement_phase(state, velocity)

        states.append(state)
        velocities.append(velocity)
    
    # book keeping
    
    states = np.array(states)
    velocities = np.array(velocities)

    mean_density = calc_density(states)
    mean_velocities = calc_velocity(velocities, states)
    flux = calc_flux(velocities)

    print("obs. density: ", np.mean(mean_density))
    print("obs. veloc: ", np.mean(mean_velocities))
    print("obs. flux: ",np.mean(flux))
    
    return states, velocities

def plot_space_time_diagram(states, rho, p, task):
    
    last = len(states)
    N = 100
    
    data_scatter = []
    
    fig, ax = plt.subplots()
    j = 1
    for i in range(last-N, last):
        
        x = np.argwhere(states[i]>0).flatten()
        y = np.ones(len(x))*j
        
        plt.scatter(x,y, s = 1, color = "black")
        
        j = j +1
        
    plt.gca().invert_yaxis()
    
    plt.xlabel("Site x", fontsize = 16)
    plt.ylabel("Timesteps t", fontsize = 16)
    
    plt.savefig("figures/" + task + "_rho=" + str(rho) + "_p=" + str(p) +".png")

def calc_flux_for_interval_rho(rhos, L, p0, p, v_max, steps, VDR = False):

    flux = []
    
    for rho in rhos:
        
        states, velocities = na_sch_model(rho, L, p0, p, v_max, steps, VDR)
        mean_flux = np.mean(calc_flux(velocities[-1000:]))
        flux.append(mean_flux)
    
    return flux

def task_3_3_6():

    L = 200
    p = 0.2

    v_max = 10

    rhos = np.arange(0.1,0.8,0.1)

    for rho in rhos:
        
        states, velocities = na_sch_model(rho, L, 0, p, v_max, 100000)
        plot_space_time_diagram(states, rho, p, "3_3_6")
        
    rho = 0.25

    p = [0.1, 0.15, 0.2, 0.25, .3]

    for prob in p:

        states, velocities = na_sch_model(rho, L, 0, prob, v_max, 100000)
        plot_space_time_diagram(states, rho, prob, "3_3_6")

def task_3_3_8():

    rhos = np.arange(0.01,0.1,0.005)
    
    rhos2 = np.arange(0.1,1,0.1)
    
    rhos = np.append(rhos, rhos2)
    
    L = 200

    p0 = 0

    v_max = 10
    steps = 10000
    
    fig, ax = plt.subplots()
    
    p = 0.1
    
    flux = calc_flux_for_interval_rho(rhos, L, p0, p, v_max, steps)

    plt.plot(rhos, flux,":s", label = "L = " + str(L) + "; p = " + str(p))
    
    p = 0.2

    flux = calc_flux_for_interval_rho(rhos, L, p0, p, v_max, steps)

    plt.plot(rhos, flux,":D", label = "L = " + str(L) + "; p = " + str(p))
    
    p = 0.3
    
    flux = calc_flux_for_interval_rho(rhos, L, p0, p, v_max, steps)

    plt.plot(rhos, flux,":o", label = "L = " + str(L) + "; p = " + str(p))

    plt.legend(fontsize = 16)
    
    plt.ylabel("Flux $J$", fontsize = 18)
    plt.xlabel("Density $\\rho$", fontsize = 18)
    
    plt.xticks(fontsize = 16)
    plt.yticks(fontsize = 16)

    plt.savefig("figures/3_3_8_fundamental_diagram.png") 
    
    # proof of convergence
    
    rho = 0.2
    steps = 1000
    
    states, velocities = na_sch_model(rho, L, p0, p, v_max, steps)
    
    mean_velocities = calc_velocity(velocities, states)
    
    fig, ax = plt.subplots()
    
    plt.plot(mean_velocities, ":D")
    
    plt.xlim([0, 50])
    
    plt.xlabel("Time steps $\Delta t$", fontsize = 18)
    plt.ylabel("Mean velocity $\langle v \\rangle $", fontsize = 18)
    
    plt.xticks(fontsize = 16)
    plt.yticks(fontsize = 16)
    
    plt.savefig("figures/3_3_8_convergence.png")

def task_3_4_8():

    L = 200
    p = 0.2
    
    p0 = 0.3

    v_max = 10

    rhos = np.arange(0.1,0.8,0.1)

    for rho in rhos:
        
        states, velocities = na_sch_model(rho, L, p0, p, v_max, 100000, VDR = True)
        plot_space_time_diagram(states, rho, p, "3_4_8")
        
    rho = 0.25

    p = [0.1, 0.15, 0.2, 0.25, .3]

    for prob in p:

        states, velocities = na_sch_model(rho, L, p0, prob, v_max, 100000,  VDR = True)
        plot_space_time_diagram(states, rho, prob, "3_4_8")

def task_3_4_9():

    rhos = np.arange(0.01,0.1,0.005)
    
    rhos2 = np.arange(0.1,1,0.1)
    
    rhos = np.append(rhos, rhos2)
    
    L = 200

    p0 = 0.35

    v_max = 10
    steps = 10000
    
    fig, ax = plt.subplots()
    
    p = 0.1
    
    flux = calc_flux_for_interval_rho(rhos, L, p0, p, v_max, steps, True)

    plt.plot(rhos, flux,":s", label = "L = " + str(L) + "; p = " + str(p) + "; $p_0 = $" + str(p0))
    
    p = 0.2

    flux = calc_flux_for_interval_rho(rhos, L, p0, p, v_max, steps, True)

    plt.plot(rhos, flux,":D", label = "L = " + str(L) + "; p = " + str(p) + "; $p_0 = $" + str(p0))
    
    p = 0.3
    
    flux = calc_flux_for_interval_rho(rhos, L, p0, p, v_max, steps, True)

    plt.plot(rhos, flux,":o", label = "L = " + str(L) + "; p = " + str(p) + "; $p_0 = $" + str(p0))

    plt.legend(fontsize = 16)
    
    plt.ylabel("Flux $J$", fontsize = 18)
    plt.xlabel("Density $\\rho$", fontsize = 18)
    
    plt.xticks(fontsize = 16)
    plt.yticks(fontsize = 16)

    plt.savefig("figures/3_4_9_fundamental_diagram.png") 
    
    # proof of convergence
    
    rho = 0.2
    steps = 1000
    
    states, velocities = na_sch_model(rho, L, p0, p, v_max, steps, True)
    
    mean_velocities = calc_velocity(velocities, states)
    
    fig, ax = plt.subplots()
    
    plt.plot(mean_velocities, ":D")
    
    plt.xlim([0, 50])
    
    plt.xlabel("Time steps $\Delta t$", fontsize = 18)
    plt.ylabel("Mean velocity $\langle v \\rangle $", fontsize = 18)
    
    plt.xticks(fontsize = 16)
    plt.yticks(fontsize = 16)
    
    plt.savefig("figures/3_4_9_convergence.png")

task_3_4_9()


