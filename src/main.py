import numpy as np 
import matplotlib.pyplot as plt 
import gym 
import copy
from tqdm import tqdm
from scipy import special


class hebbian_layers():
    def __init__(self,input_size,output_size,init_heb=False):
        self.input_size = input_size
        self.output_size = output_size
        
        self.mu = None #change rate

        self.A = None #correlation term; between pre and post synaptic activation
        self.B = None #presynaptic term
        self.C = None #postsynaptic term
        self.D = None #inhibitory/excitatory bias

        self.weights = self.init_weights()

        if(init_heb):
            self.init_hebbian_coefficients()
        
    def forward(self,input):
        output = np.matmul(self.weights,input.T)

        self.update_weights(input,output)
        return output.T

    def update_weights(self,input,output):
        #hebbian update
        presynaptic_act = np.tile(input,(self.weights.shape[0],1)) #matrix
        postsynaptic_act = np.tile(output,(self.weights.shape[1])) #matrix
        
        self.weights += self.mu*(self.A*presynaptic_act*postsynaptic_act+ self.B*presynaptic_act + self.C*postsynaptic_act + self.D)
        
        self.weight_clipping()
    
    def clipping(self,param,min_param,max_param):
        mask_min_param = param < min_param 
        param[mask_min_param] = min_param
        
        mask_max_param = param > max_param 
        param[mask_max_param] = max_param

    def weight_clipping(self,min_weights=-10,max_weights=10):
        self.clipping(self.weights,min_weights,max_weights)

    def hebbian_coefficients_clipping(self,min_coefficents=-100,max_coefficients=100):
        self.clipping(self.A,min_coefficents,max_coefficients)
        self.clipping(self.B,min_coefficents,max_coefficients)
        self.clipping(self.C,min_coefficents,max_coefficients)
        self.clipping(self.D,min_coefficents,max_coefficients)
        self.clipping(self.mu,min_coefficents,max_coefficients)


    def init_weights(self):
        return np.random.uniform(low=-0.1,high=0.1,size=(self.output_size,self.input_size))

    def init_hebbian_coefficients(self):
        self.mu = np.random.uniform(low=-0.1,high=0.1,size=(self.output_size,self.input_size))
        self.A = np.random.uniform(low=-0.1,high=0.1,size=(self.output_size,self.input_size))
        self.B = np.random.uniform(low=-0.1,high=0.1,size=(self.output_size,self.input_size))
        self.C = np.random.uniform(low=-0.1,high=0.1,size=(self.output_size,self.input_size))
        self.D = np.random.uniform(low=-0.1,high=0.1,size=(self.output_size,self.input_size))

    def load_hebbian_coefficients(self,coefficients_tuple):
        #directly assigns
        self.A, self.B, self.C, self.D, self.mu = coefficients_tuple
     
    
    def get_hebbian_coefficients(self):
        #returns a deep copy
        return (copy.deepcopy(self.A), copy.deepcopy(self.B), copy.deepcopy(self.C), copy.deepcopy(self.D), copy.deepcopy(self.mu))

    def reset(self):
        self.weights = self.init_weights()
    
    def __call__(self,input):
        return self.forward(input)

class sequential():
    def __init__(self,sequential_network):
        self.sequential_network = sequential_network

    def __call__(self,input):
        current_input = input
        for layer_act in self.sequential_network:
            output = layer_act(current_input)
            current_input = output
        return output

    def reset_all_weights(self):
        for layer_act in self.sequential_network:
            if(isinstance(layer_act,hebbian_layers)):
                layer_act.reset()

    def get_all_weights(self):
        weight_list = []
        for layer_act in self.sequential_network:
            if(isinstance(layer_act,hebbian_layers)):
                weight_list.append(layer_act.weights)
        return weight_list    
    
    def load_hebbian_coefficients(self,hebbian_coefficients_list):
        #hebbian_coefficients = (A,B,C,D,mu)
        index = 0
        for layer_act in self.sequential_network:
            if(isinstance(layer_act,hebbian_layers)):
                layer_act.load_hebbian_coefficients(hebbian_coefficients_list[index])
                index += 1

    def get_hebbian_coefficients_dict(self):
        hebbian_coefficients = []
        for layer_act in self.sequential_network:
            if(isinstance(layer_act,hebbian_layers)):
                A = layer_act.A 
                B = layer_act.B
                C = layer_act.C 
                D = layer_act.D 
                mu = layer_act.mu

                hebbian_coefficients.append({"A":A, "B":B, "C":C, "D":D,"mu":mu})

        return hebbian_coefficients    
    
    def hebbian_coefficients_clipping(self):
        for layer_act in self.sequential_network:
            if(isinstance(layer_act,hebbian_layers)):
                layer_act.hebbian_coefficients_clipping()
    
    def get_hebbian_coefficients(self):
        hebbian_coefficients = []
        for layer_act in self.sequential_network:
            if(isinstance(layer_act,hebbian_layers)):
                hebbian_coefficients.append(layer_act.get_hebbian_coefficients())

        return hebbian_coefficients    

class hebbian_agent():
    def __init__(self,input_size=128,output_size=6,hidden_size=200,init_heb=True):
        self.network = sequential([hebbian_layers(input_size,hidden_size,init_heb),np.tanh,hebbian_layers(hidden_size,hidden_size,init_heb),np.tanh,hebbian_layers(hidden_size,output_size,init_heb)])
        self.fitness = 0 
        self.action_space = output_size
        self.observation_space = input_size

    def action(self,input):
   
        output = self.network(np.resize(input,(1,self.observation_space))).flatten()

        
        #deterministic 
        chosen_action = np.argmax(output)
        
        return chosen_action
    
    def get_hebbian_coefficients(self):
        return self.network.get_hebbian_coefficients()

    def get_all_weights(self):
        return self.network.get_all_weights()

    def reset_all_weights(self):
        self.network.reset_all_weights()
    
    def load_hebbian_coefficients(self,hebbian_coefficients_list):
        self.network.load_hebbian_coefficients(hebbian_coefficients_list)
    
    def hebbian_coefficients_clipping(self):
        self.network.hebbian_coefficients_clipping()
    
    def __gt__(self,other):
        return self.fitness > other.fitness

class enviroment():
    def __init__(self,env_name="CartPole-v1"):
        self.env = gym.make(env_name)
    
    def run_episode(self,agent,amount_runs=1):
        for i in range(amount_runs):
            
            agent.reset_all_weights()

            obs = self.env.reset()
            total_reward = 0
            
            while(True):

                obs,reward,done,_ = self.env.step(agent.action(obs))
                reward = min(reward,1.0)
                

                total_reward += reward
                
                if(done):
                    break

            agent.fitness += total_reward
        agent.fitness /= amount_runs


        #render
        plt.ion()
        

        if(agent.fitness == 500):
            agent.reset_all_weights()

            obs = self.env.reset()

            fig,axis = plt.subplots(len(agent.get_all_weights()))
        

            while(True):
                for layer_num in range(len(agent.get_all_weights())):
                    axis[layer_num].imshow(agent.get_all_weights()[layer_num])
                plt.pause(0.1)
                self.env.render()

                obs,reward,done,_ = self.env.step(agent.action(obs))
                if(done):
                    break
                
            plt.close()

    def fitness_estimate(self,agent_list,epoch):
        for agent in tqdm(agent_list,ascii=True,desc="epoch "+str(epoch),unit=" agent"):
            self.run_episode(agent)


def create_new_population(agent_list,noise,lr,input_size,output_size,hidden_size):
    highest_fitness_agent = max(agent_list)
    total_fitness = np.sum([agent.fitness for agent in agent_list])
    new_population = []

    n = len(agent_list)
    sigma = noise
    alpha = lr
    
    for _ in tqdm(range(n),desc="new population",unit=" agent",ascii=True):

        hebbian_coefficients_all_layers = []
        for layer_num in range(len(highest_fitness_agent.get_hebbian_coefficients())):
            A_fitt,B_fitt,C_fitt,D_fitt,mu_fitt = highest_fitness_agent.get_hebbian_coefficients()[layer_num]
            
            delta_A = np.zeros_like(A_fitt)
            delta_B = np.zeros_like(B_fitt)
            delta_C = np.zeros_like(C_fitt)
            delta_D = np.zeros_like(D_fitt)
            delta_mu = np.zeros_like(mu_fitt) 

            for i in range(n):
                relative_fitness = agent_list[i].fitness/total_fitness
                A,B,C,D,mu = agent_list[i].get_hebbian_coefficients()[layer_num]
              
                #mutate and crossover
                delta_A += ((A-A_fitt) + sigma*np.random.normal())*relative_fitness
                delta_B += ((B-B_fitt) + sigma*np.random.normal())*relative_fitness
                delta_C += ((C-C_fitt) + sigma*np.random.normal())*relative_fitness
                delta_D += ((D-D_fitt) + sigma*np.random.normal())*relative_fitness
                delta_mu += ((mu-mu_fitt) + sigma*np.random.normal())*relative_fitness

            A_fitt += alpha*((delta_A)/n)   
            B_fitt += alpha*((delta_B)/n)   
            C_fitt += alpha*((delta_C)/n)   
            D_fitt += alpha*((delta_D)/n)   
            mu_fitt += alpha*((delta_mu)/n)  

            hebbian_coefficients_all_layers.append((A_fitt,B_fitt,C_fitt,D_fitt,mu_fitt))
        
        #make new agent
        new_agent = hebbian_agent(input_size,output_size,hidden_size,False)
        #load coefficients
        new_agent.load_hebbian_coefficients(hebbian_coefficients_all_layers)
        #clip coefficients
        new_agent.hebbian_coefficients_clipping()
        #add to population
        new_population.append(new_agent)
        

    return (new_population,total_fitness/len(agent_list),highest_fitness_agent.fitness)

def init_population(size,input_size,output_size,hidden_size):
    agent_list = []
    for i in range(size):
        agent_list.append(hebbian_agent(input_size,output_size,hidden_size))
    
    return agent_list

if __name__ == "__main__":

    #hyperparameters 
    input_size = 4
    output_size = 2
    hidden_size = 20
    EPOCHS = 1000

    #setup
    env = enviroment()

    agent_list = init_population(20,input_size,output_size,hidden_size)
    
    x_plot = []
    y_plot = []
    y_highest_plot = []

    plt.ion()
    for i in range(EPOCHS):
        env.fitness_estimate(agent_list,i)

        agent_list,average_fitness,highest_fitness = create_new_population(agent_list,noise=0.4,lr=1,input_size=input_size,output_size=output_size,hidden_size=hidden_size)
    
        x_plot.append(i)
        y_plot.append(average_fitness)
        y_highest_plot.append(highest_fitness)
        plt.plot(x_plot,y_plot,c="red",label="average_fitness")
        plt.plot(x_plot,y_highest_plot,c="blue",label="highest_fitness")

 
        plt.pause(0.1)

    plt.show()