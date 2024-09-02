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

# function, that generates a state with a certain density

@jit(nopython=True)

def create_density(rho, L):

    state = np.zeros(L)

    for i in range(int(rho*L)):

        state[i] = 1

    np.random.shuffle(state)

    return state

# function implementing the parallel update

def parallel_update(q, old_state, old_velocity):

    L = len(old_state)
    
    # copy old state, set velocity to zero

    state = np.copy(old_state)
    velocity = np.zeros(L)
    
    # get all the active agents in the system

    active_agents = np.argwhere(state>0).flatten()
    
    # iterate through the agents

    for agent in active_agents:
        
        # draw random number

        p = np.random.random_sample()
        
        # if agent is on the last site, implement periodic boundary conditions

        if agent == L-1:
            
            # can only hop if the first site on the old state is empty

            if old_state[0] == 0:

                if p < q:
                    
                    # if agent hops, change location of agent, increase velocity

                    state[agent] = 0
                    state[0] = 1
                    velocity[0] = 1

        else:
            
            # if next site is empty, agent can hop

            if old_state[agent+1] == 0:

                if p < q:
                    
                    # change location, increase velocity

                    state[agent] = 0
                    state[agent+1] = 1
                    velocity[agent+1] = 1

    return state, velocity

# parallel update function, now with open boundary conditions

def parallel_update_obc(q, old_state, old_velocity, alpha, beta):

    L = len(old_state)

    state = np.copy(old_state)
    velocity = np.zeros(L)

    active_agents = np.argwhere(state>0).flatten()

    # if site 0 is empty, an agent can enter with prob alpha

    if old_state[0] == 0:

        p = np.random.random_sample()

        if p < alpha:

            state[0] = 1
            velocity[0] = 1

    # update other agents

    for agent in active_agents:

        p = np.random.random_sample()

        if agent == L-1:
            
            # only difference, replace q with beta if agent is on last site

            if p < beta:

                state[agent] = 0

        else:

            if old_state[agent+1] == 0:

                if p < q:

                    state[agent] = 0
                    state[agent+1] = 1
                    velocity[agent+1] = 1

    return state, velocity

# function implementing the sequential update

def sequential_update(q, old_state, old_velocity):

    L = len(old_state)
    
    # copy old state, set velocity to zero

    state = np.copy(old_state)
    velocity = np.zeros(L)
    
    # get a list of active agents in descending order (np.flip)

    active_agents = np.flip(np.argwhere(state>0).flatten())

    for agent in active_agents:

        p = np.random.random_sample()
        
        # if agent is on the last site

        if agent == L-1:
            
            #now we have to look at our current state the whole time

            if state[0] == 0:

                if p < q:
                    
                    # also update current state

                    state[agent] = 0
                    state[0] = 1
                    velocity[0] = 1

        else:
            
            # if next site is empty, the agent can jump
            
            if state[agent+1] == 0:

                if p < q:
                    
                    # update current state

                    state[agent] = 0
                    state[agent+1] = 1
                    velocity[agent+1] = 1

    return state, velocity

# sequential update, this time with open boundary conditions

def sequential_update_obc(q, old_state, old_velocity, alpha, beta):

    L = len(old_state)

    state = np.copy(old_state)
    velocity = np.zeros(L)

    active_agents = np.flip(np.argwhere(state>0).flatten())
    
    # only difference is here, I add an agent which represents the possibility of an agent coming from a reservoir into the system, index -1
    
    active_agents = np.append(active_agents, -1) # -1 represents an agent coming from a reservoir
    
    # iterate through active agents

    for agent in active_agents:

        p = np.random.random_sample()
        
        # agent on last site

        if agent == L-1:
            
            # change q to beta

            if p < beta:

                state[agent] = 0
                
        # check if an agent can jump into the system
        
        elif agent == -1:
            
            # first site has to be empty

            if state[0] == 0:

                if p < alpha:

                    state[0] = 1
                    velocity[0] = 1

        else:
            
            # normal update process

            if state[agent+1] == 0:

                if p < q:

                    state[agent] = 0
                    state[agent+1] = 1
                    velocity[agent+1] = 1

    return state, velocity

# shuffle update

def shuffle_update(q, old_state, old_velocity):

    L = len(old_state)

    state = np.copy(old_state)
    velocity = np.zeros(L)

    active_agents = np.argwhere(state>0).flatten()
    
    # only difference to the sequential update is that we shuffle the array

    np.random.shuffle(active_agents)

    for agent in active_agents:

        p = np.random.random_sample()

        if agent == L-1:

            if state[0] == 0:

                if p < q:

                    state[agent] = 0
                    state[0] = 1
                    velocity[0] = 1

        else:

            if state[agent+1] == 0:

                if p < q:

                    state[agent] = 0
                    state[agent+1] = 1
                    velocity[agent+1] = 1

    return state, velocity

def shuffle_update_obc(q, old_state, old_velocity, alpha, beta):

    L = len(old_state)

    state = np.copy(old_state)
    velocity = np.zeros(L)

    active_agents = np.argwhere(state>0).flatten()
    active_agents = np.append(active_agents, -1) # -1 represents an agent coming from a reservoir
    
    # again only difference

    np.random.shuffle(active_agents)


    for agent in active_agents:

        p = np.random.random_sample()

        if agent == L-1:

            if p < beta:

                state[agent] = 0
        
        elif agent == -1:

            if state[0] == 0:

                if p < alpha:

                    state[0] = 1
                    velocity[0] = 1

        else:

            if state[agent+1] == 0:

                if p < q:

                    state[agent] = 0
                    state[agent+1] = 1
                    velocity[agent+1] = 1

    return state, velocity

# function implementing the random sequential update

def random_sequential_update(q, old_state, old_velocity):

    L = len(old_state)
    
    # copy old state

    state = np.copy(old_state)
    velocity = np.zeros(L)
    
    # get list of active agents

    active_agents = np.argwhere(state>0).flatten()

    N = len(active_agents)
    
    # set number of iterations

    for i in range(N):
        
        #pick a random agent from active agents
        
        agent_index =  np.random.randint(0,len(active_agents))
        
        # get index of agent
        
        agent = active_agents[agent_index]

        p = np.random.random_sample()
        
        # if agent is on last site, do this

        if agent == L-1:

            if state[0] == 0:

                if p < q:

                    state[agent] = 0
                    state[0] = 1
                    velocity[0] = 1
                    
                    # we now have to update our list of active agents
                    
                    active_agents = np.delete(active_agents, agent_index)
                    active_agents = np.append(active_agents, 0)

        else:

            if state[agent+1] == 0:

                if p < q:

                    state[agent] = 0
                    state[agent+1] = 1
                    velocity[agent+1] = 1
                    
                    # again update list of active agents
                    
                    active_agents = np.delete(active_agents, agent_index)
                    active_agents = np.append(active_agents, agent+1)


    return state, velocity

# random sequential update with open boundary conditions

@jit(nopython=True)

def random_sequential_update_obc(q, old_state, old_velocity, alpha, beta):

    L = len(old_state)
    
    # copy states

    state = np.copy(old_state)
    velocity = np.zeros(L)
    
    # compile list of active agents

    active_agents = np.argwhere(state>0).flatten()

    N = len(active_agents)
    
    # add the possibility of an agent coming from the reservoir

    active_agents = np.append(active_agents, -1) # -1 represents an agent coming from a reservoir

    for i in range(N):
        
        # again, pick a random agent

        agent_index =  np.random.randint(0,len(active_agents))
        
        agent = active_agents[agent_index]

        p = np.random.random_sample()
        
        # check if its on the last site

        if agent == L-1:

            if p < beta:

                state[agent] = 0
                
                active_agents = np.delete(active_agents, agent_index)
                
        # check if an agent can jump into the system
        
        elif agent == -1:

            if state[0] == 0:

                if p < alpha:

                    state[0] = 1
                    velocity[0] = 1
                    
                    # we dont have to delete -1 from active agents, since an element can hop in at any time
                    #active_agents = np.delete(active_agents, agent_index) 
                    
                    active_agents = np.append(active_agents, agent+1)

        else:
            
            # normal update process

            if state[agent+1] == 0:

                if p < q:

                    state[agent] = 0
                    state[agent+1] = 1
                    velocity[agent+1] = 1
                    
                    active_agents = np.delete(active_agents, agent_index)
                    active_agents = np.append(active_agents, agent+1)

    return state, velocity


def test_parallel_update():

    working = True

    state, velocity = parallel_update(1, [0, 1, 0, 0, 1], [], True)

    if not np.array_equal(state, [1,0,1,0,0]) or not np.array_equal(velocity,[1,0,1,0,0]):

        working = False

    state, velocity = parallel_update(1, [0, 1, 1, 0, 1], [], True)
    
    if not np.array_equal(state, [1,1,0,1,0]) or not np.array_equal(velocity,[1,0,0,1,0]):

        working = False

    print("test: ", working)

# calculates the density

def calc_density(states):

    return np.mean(states, axis = 0)

# calculates the velocity

@jit(nopython=True)

def calc_velocity(velocities, states):
    
    M = len(velocities)
    
    mean_veloc = []
    
    for i in range(M):
        
        state = states[i]
        velocity = velocities[i]
        
        velocity = velocity[state > 0]
        
        if len(velocity)==0:
            
            mean_veloc.append(0)
            
        else:
            
            mean_veloc.append(np.mean(velocity))

    return mean_veloc

# calculates the flux

def calc_flux(velocities):

    return np.mean(velocities, axis = 0)

# calculates density + flux of the bulk

def calc_density_flux_of_bulk(states, velocities):

    L = len(velocities[0])

    mean_density = []
    mean_velocities = []

    for i in range(len(velocities)):

        veloc = velocities[i]
        state = states[i]
        
        mean_veloc = np.mean(veloc[int(0.2*L):int(0.8*L)])

        density = np.mean(state[int(0.2*L):int(0.8*L)])

        mean_velocities.append(mean_veloc)
        mean_density.append(density)
    
    return mean_density, mean_velocities


def binning_analysis(mean_velocities, bin_size):

    max_bin = int(len(mean_velocities)/bin_size)*bin_size

    binned = np.reshape(mean_velocities[0:max_bin], (-1, bin_size))

    avg_binned = np.mean(binned, axis = 1)

    x = np.arange(bin_size/2, len(mean_velocities), bin_size)

    return x, avg_binned


# implementation of the ASEP with periodic boundary conditions
# update method is handed over as an argument

def ASEP_PBC(q, rho, L, steps, update_scheme):
    
    # create starting state

    state = create_density(rho, L)
    velocity = np.zeros(L)

    states = [state]
    velocities = [velocity]

    for i in range(steps):
        
        # get new state + new velocities from the update method
        # application of different update methods, depending on which update method is handed over

        state, velocity = update_scheme(q, state, velocity)

        states.append(state)
        velocities.append(velocity)

    states = np.array(states)
    velocities = np.array(velocities)


    mean_density = calc_density(states)
    mean_velocities = calc_velocity(velocities, states)
    flux = calc_flux(velocities)

    print("obs. density: ", np.mean(mean_density), "; theor. density: ", rho)
    print("obs. veloc: ", np.mean(mean_velocities),"; theor. veloc: ", q*(1-rho))
    print("obs. flux: ",np.mean(flux),"; theor. flux: ", q*rho*(1-rho))
    
    return states, velocities

def plot_space_time_diagram(states, L, q, update_method):
    
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
    
    plt.savefig("figures/3_1_1_q=" + str(q) +"_" + update_method +".png")
        
        
def calc_flux_for_interval_rho(q, rhos, L, parallel_update):
    
    flux = []
    
    for rho in rhos:
        
        states, velocities = ASEP_PBC(q, rho, L, 10000, parallel_update)
        mean_flux = np.mean(calc_flux(velocities[-1000:]))
        flux.append(mean_flux)
    
    return flux


# implementation of the ASEP with open boundary conditions
# update method is handed over as an argument

def ASEP_OBC(q, rho, L, alpha, beta, steps, update_scheme):

    state = create_density(rho, L)
    velocity = np.zeros(L)

    states = [state]
    velocities = [velocity]

    for i in range(steps):
        
        # application of different update methods, depending on which update method is handed over

        state, velocity = update_scheme(q, state, velocity, alpha, beta)

        states.append(state)
        velocities.append(velocity)
        
    # do some book keeping

    states = np.asarray(states)
    velocities = np.asarray(velocities)

    mean_density = calc_density(states)
    mean_velocities = calc_velocity(velocities, states)
    flux = calc_flux(velocities)

    print("obs. density: ", np.mean(mean_density), "; theor. density: ", rho)
    print("obs. veloc: ", np.mean(mean_velocities),"; theor. veloc: ", q*(1-rho))
    print("obs. flux: ",np.mean(flux),"; theor. flux: ", q*rho*(1-rho))
    
    return states, velocities


def plot_proof_of_convergence(q, rho, alpha, beta, states, velocities):

    fig, ax = plt.subplots()

    density, flux = calc_density_flux_of_bulk(states, velocities)

    x, binned = binning_analysis(flux[:5000], 20)

    fig, ax = plt.subplots()

    plt.plot(x, binned, label = "$q = $" + str(q) + "; $\\rho_0 = $" + str(rho) + "; $\\alpha = $" + str(alpha) + "; $\\beta=$" + str(beta))

    plt.xlabel("Steps", fontsize = 16)
    plt.ylabel("Flux $J$", fontsize = 16)

    plt.legend(fontsize = 16)

    plt.savefig("figures/3_2_3_converging_flux_alpha=" + str(alpha) + "_beta=" + str(beta) + ".png")
    

def task_3_1_1():   
    
    L = 200
    q = 1
    rho = 0.5
    
    states, velocities = ASEP_PBC(q, rho, L, 100000, parallel_update)
    
    plot_space_time_diagram(states, L, q, "parallel_update")
    
    states, velocities = ASEP_PBC(q, rho, L, 100000, sequential_update)
    
    plot_space_time_diagram(states, L, q, "sequential_update")
    
    states, velocities = ASEP_PBC(q, rho, L, 100000, shuffle_update)
    
    plot_space_time_diagram(states, L, q, "shuffle_update")
    
    states, velocities = ASEP_PBC(q, rho, L, 100000, random_sequential_update)
    
    plot_space_time_diagram(states, L, q, "random_sequential_update")
    
    L = 200
    q = 0.5
    rho = 0.5
    
    states, velocities = ASEP_PBC(q, rho, L, 100000, parallel_update)
    
    plot_space_time_diagram(states, L, q, "parallel_update")
    
    states, velocities = ASEP_PBC(q, rho, L, 100000, sequential_update)
    
    plot_space_time_diagram(states, L, q, "sequential_update")
    
    states, velocities = ASEP_PBC(q, rho, L, 100000, shuffle_update)
    
    plot_space_time_diagram(states, L, q, "shuffle_update")
    
    states, velocities = ASEP_PBC(q, rho, L, 100000, random_sequential_update)
    
    plot_space_time_diagram(states, L, q, "random_sequential_update")

def task_3_1_2():
    
    rhos = np.arange(0.1, 1, 0.1)
    q = 0.5
    L = 200
    
    theory = q*rhos*(1-rhos)
    
    fig, ax = plt.subplots()
    
    flux = calc_flux_for_interval_rho(q, rhos, L, parallel_update)
    
    plt.plot(rhos, flux, ":D",label = "q = " + str(q) + "; L = " + str(L) + "; parallel update")
    
    flux = calc_flux_for_interval_rho(q, rhos, L, sequential_update)
    
    plt.plot(rhos, flux, ":s",label = "q = " + str(q) + "; L = " + str(L) + "; sequential update")
    
    flux = calc_flux_for_interval_rho(q, rhos, L, shuffle_update)
    
    plt.plot(rhos, flux, ":H",label = "q = " + str(q) + "; L = " + str(L) + "; shuffle update")
    
    flux = calc_flux_for_interval_rho(q, rhos, L, random_sequential_update)
    
    plt.plot(rhos, flux, ":o",label = "q = " + str(q) + "; L = " + str(L) + "; rand. seq. update")
    
    # plot theory
    rhos = np.linspace(0, 1, 100)
    theory = q*rhos*(1-rhos)
    
    plt.plot(rhos, theory, "-.",label = " Theory $J(\\rho) = q \\rho (1-\\rho)$")
    
    plt.legend(fontsize = 16)
    
    plt.savefig("figures/3_1_2_q=" + str(q) + ".png")
    
    #########################
    
    rhos = np.arange(0.1, 1, 0.1)
    q = 0.7
    L = 200
    
    theory = q*rhos*(1-rhos)
    
    fig, ax = plt.subplots()
    
    flux = calc_flux_for_interval_rho(q, rhos, L, parallel_update)
    
    plt.plot(rhos, flux, ":D",label = "q = " + str(q) + "; L = " + str(L) + "; parallel update")
    
    flux = calc_flux_for_interval_rho(q, rhos, L, sequential_update)
    
    plt.plot(rhos, flux, ":s",label = "q = " + str(q) + "; L = " + str(L) + "; sequential update")
    
    flux = calc_flux_for_interval_rho(q, rhos, L, shuffle_update)
    
    plt.plot(rhos, flux, ":H",label = "q = " + str(q) + "; L = " + str(L) + "; shuffle update")
    
    flux = calc_flux_for_interval_rho(q, rhos, L, random_sequential_update)
    
    plt.plot(rhos, flux, ":o",label = "q = " + str(q) + "; L = " + str(L) + "; rand. seq. update")
    
    # plot theory
    rhos = np.linspace(0, 1, 100)
    theory = q*rhos*(1-rhos)
    
    plt.plot(rhos, theory, "-.",label = " Theory $J(\\rho) = q \\rho (1-\\rho)$")
    
    plt.legend(fontsize = 16)
    
    plt.savefig("figures/3_1_2_q=" + str(q) + ".png")
    
    # proof of convergence
    
    q = 0.7
    rho = 0.5
    L = 200
    
    states, velocities = ASEP_PBC(q, rho, L, 10000, random_sequential_update)
    
    mean_velocities = calc_velocity(velocities, states)
    
    fig, ax = plt.subplots()
    
    plt.plot(mean_velocities, ":D", label = "L = " + str(L) +"; $ q = $" + str(q) + "; $\\rho = $" + str(rho))
    
    plt.xlim([0, 50])
    
    plt.xlabel("Time steps $\Delta t$", fontsize = 18)
    plt.ylabel("Mean velocity $\langle v \\rangle $", fontsize = 18)
    
    plt.legend(fontsize = 16)
    
    plt.xticks(fontsize = 16)
    plt.yticks(fontsize = 16)
    
    plt.savefig("figures/3_1_2_convergence.png")

def task_3_2_3():

    L = 200

    steps = 100000

    q = 0.8
    rho = 0.01

    alpha = 0.7
    beta = 0.7

    states, velocities = ASEP_OBC( q, rho, L, alpha, beta, steps, random_sequential_update_obc)

    fig, ax = plt.subplots()

    density = calc_density(states)

    plt.plot(density, label = "$\\alpha = $" + str(alpha) + "; $\\beta = $" + str(beta) + "; steps = " + str(steps))

    plt.legend(fontsize = 16)

    plt.xlabel("Site $x_i$", fontsize = 16)
    plt.ylabel("Density $\\rho$ at $x_i$", fontsize = 16)

    plt.savefig("figures/3_2_3_density_profile_alpha=" + str(alpha) + "_beta=" + str(beta) + ".png")

    plot_proof_of_convergence(q, rho, alpha, beta, states, velocities)

    alpha = 0.5
    beta = 0.25

    states, velocities = ASEP_OBC( q, rho, L, alpha, beta, steps, random_sequential_update_obc)

    fig, ax = plt.subplots()

    density = calc_density(states)

    plt.plot(density, label = "$\\alpha = $" + str(alpha) + "; $\\beta = $" + str(beta) + "; steps = " + str(steps))

    plt.legend(fontsize = 16)

    plt.xlabel("Site $x_i$", fontsize = 16)
    plt.ylabel("Density $\\rho$ at $x_i$", fontsize = 16)

    plt.savefig("figures/3_2_3_density_profile_alpha=" + str(alpha) + "_beta=" + str(beta) + ".png")

    plot_proof_of_convergence(q, rho, alpha, beta, states, velocities)

    alpha = 0.8
    beta = 0.25

    states, velocities = ASEP_OBC( q, rho, L, alpha, beta, steps, random_sequential_update_obc)

    fig, ax = plt.subplots()

    density = calc_density(states)

    plt.plot(density, label = "$\\alpha = $" + str(alpha) + "; $\\beta = $" + str(beta) + "; steps = " + str(steps))

    plt.legend(fontsize = 16)

    plt.xlabel("Site $x_i$", fontsize = 16)
    plt.ylabel("Density $\\rho$ at $x_i$", fontsize = 16)

    plt.savefig("figures/3_2_3_density_profile_alpha=" + str(alpha) + "_beta=" + str(beta) + ".png")

    plot_proof_of_convergence(q, rho, alpha, beta, states, velocities)

    alpha = 0.25
    beta = 0.5

    states, velocities = ASEP_OBC( q, rho, L, alpha, beta, steps, random_sequential_update_obc)

    fig, ax = plt.subplots()

    density = calc_density(states)

    plt.plot(density, label = "$\\alpha = $" + str(alpha) + "; $\\beta = $" + str(beta) + "; steps = " + str(steps))

    plt.legend(fontsize = 16)

    plt.xlabel("Site $x_i$", fontsize = 16)
    plt.ylabel("Density $\\rho$ at $x_i$", fontsize = 16)

    plt.savefig("figures/3_2_3_density_profile_alpha=" + str(alpha) + "_beta=" + str(beta) + ".png")

    plot_proof_of_convergence(q, rho, alpha, beta, states, velocities)

    alpha = 0.25
    beta = 0.8

    states, velocities = ASEP_OBC( q, rho, L, alpha, beta, steps, random_sequential_update_obc)

    fig, ax = plt.subplots()

    density = calc_density(states)

    plt.plot(density, label = "$\\alpha = $" + str(alpha) + "; $\\beta = $" + str(beta) + "; steps = " + str(steps))

    plt.legend(fontsize = 16)

    plt.xlabel("Site $x_i$", fontsize = 16)
    plt.ylabel("Density $\\rho$ at $x_i$", fontsize = 16)

    plt.savefig("figures/3_2_3_density_profile_alpha=" + str(alpha) + "_beta=" + str(beta) + ".png")

    plot_proof_of_convergence(q, rho, alpha, beta, states, velocities)

def task_3_2_4():

    L = 200

    steps = 10000

    q = 0.8
    rho = 0.01

    alpha = np.arange(0.05, 1, 0.05)
    beta = np.arange(0.05, 1, 0.05)

    density = np.zeros((len(beta),len(alpha)))
    flux = np.zeros((len(beta),len(alpha)))

    for i in range(len(beta)):

        for j in range(len(alpha)):

            states, velocities = ASEP_OBC( q, rho, L, alpha[j], beta[i], steps, random_sequential_update_obc)

            density_array, flux_array = calc_density_flux_of_bulk(states, velocities)

            density[i, j] = np.mean(density_array)
            flux[i, j] = np.mean(flux_array)
    
    fig, ax = plt.subplots(layout = "constrained")

    cf = ax.contourf(alpha, beta, density)
    
    ax.contour(alpha, beta, density, levels = [0.45, 0.55])

    plt.xlabel("$\\alpha$", fontsize = 16)
    plt.ylabel("$\\beta$", fontsize = 16)

    plt.title("Density $\\rho$ depending on $\\alpha$ and $\\beta$", fontsize = 16)

    fig.colorbar(cf, ax = ax)

    plt.savefig("figures/3_2_4_density.png")

    fig, ax = plt.subplots(layout = "constrained")

    cf = ax.contourf(alpha, beta, flux)

    plt.xlabel("$\\alpha$", fontsize = 16)
    plt.ylabel("$\\beta$", fontsize = 16)

    plt.title("Flux $J$ depending on $\\alpha$ and $\\beta$", fontsize = 16)

    fig.colorbar(cf, ax = ax)

    plt.savefig("figures/3_2_4_flux.png")

def task_3_2_5():

    L = 200

    steps = 10000

    q = 0.8
    rho = 0.01

    alpha = np.arange(0.05, 0.95, 0.01)
    beta = np.arange(0.05, 0.95, 0.01)

    shape = np.zeros((len(beta),len(alpha)))

    for i in range(len(beta)):

        for j in range(len(alpha)):

            states, velocities = ASEP_OBC( q, rho, L, alpha[j], beta[i], steps, random_sequential_update_obc)

            density_profile = calc_density(states)

            shape[i, j] = density_profile[-1]-density_profile[0]
    
    fig, ax = plt.subplots(layout = "constrained")

    cf = ax.contourf(alpha, beta, shape)

    plt.xlabel("$\\alpha$", fontsize = 18)
    plt.ylabel("$\\beta$", fontsize = 18)
    
    plt.xticks(fontsize = 16)
    plt.yticks(fontsize = 16)

    plt.title("Shape parameter $\Delta\\rho$ depending on $\\alpha$ and $\\beta$", fontsize = 16)

    cbar = fig.colorbar(cf, ax = ax)
    cbar.ax.tick_params(labelsize=16)

    plt.savefig("figures/3_2_5_shape.png")

task_3_1_2()

