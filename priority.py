import numpy as np
import torch
import random
from torch import nn
from graph import *
import copy

## Network Parameters ############
BATCH_SIZE = 64
##################################

## Game Parameters ###############
PLAYER_RED = -1
PLAYER_BLUE = 1
DRAW = 0

VERTICES = 16
##################################

## Reward Structure ##############
WIN_REWARD = 100
MOVE_REWARD = 1
DRAW_REWARD = 0
##################################

def get_batch(data, size, networks, current_player):
    other_player = current_player * -1
    l = [i for i in range(len(data[current_player])) ]
    
    priorities = [ data[current_player][i][-1] for i in range( len(data[current_player] ) ) ]
    
    #priorities = data[current_player][:][-1]
    #print( len( priorities ) )
    #print(type(priorities))
    #print(type(priorities[0]))
    #print(priorities[0].shape)
    #print(priorities[0])
    #print( len( data[current_player][0] ))
    #print(data[current_player][0][-1])
    #print( len( data[current_player][:] ) )
    indices = random.choices( l, weights = priorities, k = BATCH_SIZE )
    #indices = random.sample(range(0, len(data[current_player])-1), size)
    networks[ other_player ].eval()
    
    records = []
    for i in range(size):
        records.append( data[current_player][ indices[i] ] )
    
    input_tensor = torch.empty( size = (size, 2, VERTICES, VERTICES) )
    output_tensor = torch.empty( size = (size, 1) )
    
    for i in range(size):
        current_state, action, next_state, reward, game_over, loss = records[i]
        
        input_tensor[i, 0, :, :] = torch.tensor( current_state )
        input_tensor[i, 1, :, :] = torch.tensor( action )
        
        max_q = None
        
        if not game_over:
            
            possible_actions = get_possible_actions(next_state)
            mega_states = []
            for action in possible_actions:
                test_action = np.zeros((VERTICES,VERTICES))
                test_action[action[0], action[1]] = other_player
                test_action[action[1], action[0]] = other_player
                
                mega_states.append( np.array([next_state, test_action]) )
            mega_states = np.array( mega_states )
            mega_input = torch.tensor( mega_states ).float()
            max_q = torch.max( networks[ other_player ]( mega_input ) ).detach()
                            
        else:
            max_q = 0
        
        output_tensor[i, 0] = reward - 0.9 * max_q

    return input_tensor, output_tensor
                
class QNN(nn.Module):
    def __init__(self):
        super(QNN, self).__init__()

        self.layer_1 = nn.Conv2d(in_channels=2, out_channels=100, kernel_size=(1,VERTICES), stride=1)
        self.dropout = nn.Dropout(p=0.2)
        self.flatten = nn.Flatten()
        self.layer_2 = nn.Linear(in_features=VERTICES*100, out_features=100, bias=True)
        self.layer_3 = nn.Linear(in_features=100, out_features=100, bias=True)
        self.layer_4 = nn.Linear(in_features=100, out_features=1, bias=True)
        
    def forward(self, input_tensor):
        output = input_tensor
        output = self.layer_1(output)
        output = self.flatten(output)
        output = nn.ELU()(output)
        output = self.dropout(output)
        output = self.layer_2(output)
        output = nn.ELU()(output)
        output = self.layer_3(output)
        output = nn.ELU()(output)
        output = self.layer_4(output)
        return output

def train_one_pass( data, actors, critics, current_player, optimizers ):
    loss = nn.MSELoss()
    other_player = current_player * -1
    critics[ current_player ].train()
    
    total_loss = 0.0
    n = len( data[ current_player ] )
    for i in range( n // BATCH_SIZE ):
        batch_x, batch_y = get_batch( data, BATCH_SIZE, actors, current_player  )
        batch_x = batch_x.detach()
        batch_y = batch_y.detach()
        
        output = critics[ current_player ]( batch_x )
        current_loss = loss( output, batch_y )
        
        optimizers[ current_player ].zero_grad()
        current_loss.backward()
        optimizers[ current_player ].step()
        total_loss += current_loss.detach().numpy()

    #print(len(data[current_player]))
    n_input = np.array([[data[current_player][i][0], data[current_player][i][1]] for i in range(len(data[current_player]))])
    #print(len(n_input), len(n_input[0]))
    eval = actors[current_player](torch.tensor(n_input).float()).detach().numpy()
    for i in range(len(data[current_player])):
        data[current_player][i][-1] = (eval[i] - data[current_player][i][-3])**2
    
    return np.sqrt( total_loss / ( n // BATCH_SIZE ) )

def get_bern(p):
    rand = np.random.random()
    if rand <= p:
        return 1
    else: 
        return 0 

def get_possible_actions(adj_mat):
    possible_actions = []
    for a in range(VERTICES):
        for b in range(a+1, VERTICES):
            if adj_mat[a, b] == 0:
                possible_actions.append((a,b))
    return possible_actions

def get_network_move(network, curr_state, player):
    network.eval()
    possible_actions = get_possible_actions(curr_state)
    if len( possible_actions ) == 0:
        print("What happened?")
        input()
    
    mega_states = []
    for action in possible_actions:
        action_mat = np.zeros((VERTICES, VERTICES))
        action_mat[action[0], action[1]] = player
        action_mat[action[1], action[0]] = player
        mega_states.append( np.array([curr_state, action_mat]) )
    mega_states = np.array( mega_states )
    mega_input = torch.tensor( mega_states ).float()
    mega_curr_q = network( mega_input )
    
    max_q = torch.max( mega_curr_q ).detach()
    
    best_actions = []
    
    for i in range( len( mega_states ) ):
        if mega_curr_q[ i, 0 ] == max_q:
            best_actions.append( possible_actions[i] )

    if len( best_actions ) == 0:
        print("What?")
        input()
    
    return random.choice( best_actions )

def get_random_move(curr_state, player):
    possible_actions = get_possible_actions(curr_state)
    rand = round(np.random.random()*(len(possible_actions)-1))
    return possible_actions[rand]

def get_move(network, curr_state, player, probability):
    rand = get_bern(probability)
    if rand == 1: 
        move = get_network_move(network, curr_state, player)
    else: 
        move = get_random_move(curr_state, player)    
    
    return move

def initialize_game():
    adj_mat = np.zeros((VERTICES, VERTICES))
    
    vert_list = [ i for i in range( VERTICES ) ]
    red_x, red_y = random.sample( vert_list, k = 2 )
    if red_x == red_y:
        print("Ooops")
        input()
    vert_list.remove( red_x )
    vert_list.remove( red_y )
    
    if get_bern( 0.5 ) == 1:
        blue_x = red_x
        blue_y = random.choice( vert_list )
    else:
        blue_x, blue_y = random.sample( vert_list, k = 2 )
    
    adj_mat[ red_x, red_y ] = PLAYER_RED
    adj_mat[ red_y, red_x ] = PLAYER_RED
    adj_mat[ blue_x, blue_y ] = PLAYER_BLUE
    adj_mat[ blue_y, blue_x ] = PLAYER_BLUE
    
    return adj_mat

def play_game(networks, data, probabilities):
    win_message = { PLAYER_RED : "Player Red Won!", PLAYER_BLUE : "Player Blue Won!" }
    adj_mat = initialize_game()
    remaining_edges = len(get_possible_actions(adj_mat))
    
    new_edge = None
    current_player = PLAYER_RED
    
    while remaining_edges > 0:
        current_state = np.zeros((VERTICES,VERTICES))
        current_state += adj_mat
        
        new_edge = get_move( networks[current_player], adj_mat, current_player, probabilities[ current_player ] )
        
        action = np.zeros((VERTICES,VERTICES))
        action[new_edge[0]][new_edge[1]] = current_player
        action[new_edge[1]][new_edge[0]] = current_player
        
        next_state = np.zeros((VERTICES,VERTICES))
        next_state += current_state
        next_state[new_edge[0], new_edge[1]] = current_player
        next_state[new_edge[1], new_edge[0]] = current_player
        
        adj_mat[new_edge[0], new_edge[1]] = current_player
        adj_mat[new_edge[1], new_edge[0]] = current_player
        remaining_edges -= 1

        eval = networks[current_player](torch.tensor(np.array([[current_state, action]])).float()).detach().numpy()[0,0]
        #print( eval.shape )
        
        if clique_4(adj_mat, new_edge, current_player):
            record = [ current_state, action, next_state, WIN_REWARD, True, (eval-WIN_REWARD)**2 ]
            data[ current_player ].append( record )
            
            return current_player, remaining_edges
        else:
            
            if remaining_edges == 0:
                record = [ current_state, action, next_state, DRAW_REWARD, True, (eval-DRAW_REWARD)**2 ]
                data[ current_player ].append( record )
                return 0, remaining_edges
            else:
                record = [ current_state, action, next_state, MOVE_REWARD, False, (eval-MOVE_REWARD)**2 ]
                data[ current_player ].append( record )
                current_player *= -1        
    
    print("Literally should never land here.")
    return None, None

def save_current_state( actors, critics, optimizers ):
    torch.save( critics[PLAYER_RED].state_dict(), "critic_red" )
    torch.save( critics[PLAYER_BLUE].state_dict(), "critic_blue" )
    
    torch.save( actors[PLAYER_RED].state_dict(), "actor_red" )
    torch.save( actors[PLAYER_BLUE].state_dict(), "actor_blue" )
    
    torch.save( optimizers[PLAYER_RED].state_dict(), "opt_red" )
    torch.save( optimizers[PLAYER_BLUE].state_dict(), "opt_blue" )

def load_current_state( actors, critics, optimizers ):
    critics[ PLAYER_RED ].load_state_dict( torch.load( "critic_red" ) )
    critics[ PLAYER_BLUE ].load_state_dict( torch.load( "critic_blue" ) )
    
    actors[ PLAYER_RED ].load_state_dict( torch.load( "actor_red" ) )
    actors[ PLAYER_BLUE ].load_state_dict( torch.load( "actor_blue" ) )
    
    optimizers[ PLAYER_RED ].load_state_dict( torch.load( "opt_red" ) )
    optimizers[ PLAYER_BLUE ].load_state_dict( torch.load( "opt_blue" ) )

def evaluate_networks( networks, probabilities ):
    temp_data = { PLAYER_RED : [], PLAYER_BLUE : [] }
    
    test_wins = { PLAYER_RED : 0, PLAYER_BLUE : 0, DRAW : 0 }
    length_data = []
    print("Loading data for 100 games")
    for i in range(100):
        winner, remaining_edges = play_game( networks, temp_data, probabilities )
        test_wins[ winner ] += 1
        length_data.append( remaining_edges )
    
    print("Current Data")
    print("\tBlue:", test_wins[ PLAYER_BLUE ] )
    print("\tRed: ", test_wins[ PLAYER_RED ] )
    print("\tDraw: ", test_wins[ DRAW ] )
    print("\tGame Remaining: ", sum( length_data ) / 100.0 )
    print()
    
    return temp_data, test_wins, length_data

if __name__ == "__main__":
    
    ##########################################################################
    actor_red = QNN()
    actor_blue = QNN()
    
    critic_red = QNN()
    critic_blue = QNN()
    
    optimizer_red = torch.optim.Adam(critic_red.parameters(), lr=1e-3)
    optimizer_blue = torch.optim.Adam(critic_blue.parameters(), lr=1e-3)
    
    actors = { PLAYER_RED : actor_red, PLAYER_BLUE : actor_blue }
    critics = { PLAYER_RED : critic_red, PLAYER_BLUE : critic_blue }
    optimizers = { PLAYER_RED : optimizer_red, PLAYER_BLUE : optimizer_blue }
    
    load_current_state( actors, critics, optimizers )
    
    ##########################################################################
    random_prob     = { PLAYER_RED : 0.00, PLAYER_BLUE : 0.00 }
    selection_prob  = { PLAYER_RED : 0.90, PLAYER_BLUE : 0.90 }
    evaluation_prob = { PLAYER_RED : 1.00, PLAYER_BLUE : 1.00 }
    ##########################################################################
    
    data, test_wins, length_data = evaluate_networks( actors, evaluation_prob )

    for mega_rounds in range( 100 ):
        print("\n***** Mega-Round", mega_rounds, "*****")
        for rounds in range( 10 ):
            print("\n*** Round", rounds, "***")
            data, test_wins, length_data = evaluate_networks( actors, selection_prob )
            data_red = data[ PLAYER_RED ]
            data_blue = data[ PLAYER_BLUE ]
    
            if test_wins[ DRAW ] == 100 :
                print("Gottem. Breaking early.")
                break
    
            print( "Data for Red Player:", len( data_red ) )
            print( "Data for Blue Player:", len( data_blue ) )
        
            for i in range(10):
                avg_red_loss = train_one_pass(data, actors, critics, PLAYER_RED,  optimizers)
                avg_blue_loss = train_one_pass(data, actors, critics, PLAYER_BLUE, optimizers)
            
                print(i, "\t(Red/Blue): \t", avg_red_loss, "\t\t", avg_blue_loss)
    
    
            ########################################################################
            if not np.isnan( avg_red_loss ) and not np.isnan( avg_blue_loss ):
                print("Saving States")
                save_current_state( actors, critics, optimizers )
            ########################################################################

        # if mega_rounds % 2 == 0:
        #     actors[PLAYER_RED] = copy.deepcopy( critics[PLAYER_RED] )
        # else:
        #     actors[PLAYER_BLUE] = copy.deepcopy( critics[PLAYER_BLUE] )
        actors[PLAYER_RED] = copy.deepcopy( critics[PLAYER_RED] )
        actors[PLAYER_BLUE] = copy.deepcopy( critics[PLAYER_BLUE] )
    
        save_current_state( actors, critics, optimizers )

    data, test_wins, length_data = evaluate_networks( actors, evaluation_prob )
