'''
Created on Jan. 16, 2021

@author: zollen
'''

import random
from operator import itemgetter

class GreenBerb:
    
    TRIALS = 1000
    
    def __init__(self):
        self.rps_to_text = ('rock','paper','scissors')
        self.rps_to_num  = {'rock':0, 'paper':1, 'scissors':2}
        self.wins_with = (1,2,0)      #superior
        self.best_without = (2,0,1)   #inferior
        self.lengths = (10, 20, 30, 40, 49, 0)
        self.opponent_hist = []
        self.my_hist = []
        self.score_table = ((0,-1,1),(1,0,-1),(-1,1,0))
        self.p_random_score = 0
        self.act = None
        
    def min_index(self, values):
        return min(enumerate(values), key=itemgetter(1))[0]
    
    def max_index(self, values):
        return max(enumerate(values), key=itemgetter(1))[0]
    
    def find_best_prediction(self, l, p_random):  # l = len
        bs = -self.TRIALS
        bp = 0
        T = len(self.opponent_hist)  #so T is number of trials completed
        if self.p_random_score > bs:
            bs = self.p_random_score
            bp = p_random
        for i in range(3):
            for j in range(24):
                for k in range(4):
                    new_bs = self.p_full_score[T%50][j][k][i] - (self.p_full_score[(50+T-l)%50][j][k][i] if l else 0)
                    if new_bs > bs:
                        bs = new_bs
                        bp = (self.p_full[j][k] + i) % 3
                for k in range(2):
                    new_bs = self.r_full_score[T%50][j][k][i] - (self.r_full_score[(50+T-l)%50][j][k][i] if l else 0)
                    if new_bs > bs:
                        bs = new_bs
                        bp = (self.r_full[j][k] + i) % 3
            for j in range(2):
                for k in range(2):
                    new_bs = self.p_freq_score[T%50][j][k][i] - (self.p_freq_score[(50+T-l)%50][j][k][i] if l else 0)
                    if new_bs > bs:
                        bs = new_bs
                        bp = (self.p_freq[j][k] + i) % 3
                    new_bs = self.r_freq_score[T%50][j][k][i] - (self.r_freq_score[(50+T-l)%50][j][k][i] if l else 0)
                    if new_bs > bs:
                        bs = new_bs
                        bp = (self.r_freq[j][k] + i) % 3
        return bp
    
    def decide(self):
        
        p_random = random.choice([0,1,2])  #called 'guess' in iocaine
        T = len(self.opponent_hist)  #so T is number of trials completed
    
        if not self.my_hist:
            self.opp_history = [0]  #pad to match up with 1-based move indexing in original
            self.my_history = [0]
            self.gear = [[0] for _ in range(24)]
            # init()
            self.p_random_score = 0
            self.p_full_score = [[[[0 for i in range(3)] for k in range(4)] for j in range(24)] for _ in range(50)]
            self.r_full_score = [[[[0 for i in range(3)] for k in range(2)] for j in range(24)] for _ in range(50)]
            self.p_freq_score = [[[[0 for i in range(3)] for k in range(2)] for j in range(2)] for _ in range(50)]
            self.r_freq_score = [[[[0 for i in range(3)] for k in range(2)] for j in range(2)] for _ in range(50)]
            self.s_len = [0] * 6
    
            self.p_full = [[0,0,0,0] for _ in range(24)]
            self.r_full = [[0,0] for _ in range(24)]
        else:
            self.my_history.append(self.rps_to_num[self.my_hist[-1]])
            self.opp_history.append(self.rps_to_num[self.opponent_hist[-1]])
            # update_scores()
            self.p_random_score += self.score_table[p_random][self.opp_history[-1]]
            self.p_full_score[T%50] = [[[self.p_full_score[(T+49)%50][j][k][i] + self.score_table[(self.p_full[j][k] + i) % 3][self.opp_history[-1]] for i in range(3)] for k in range(4)] for j in range(24)]
            self.r_full_score[T%50] = [[[self.r_full_score[(T+49)%50][j][k][i] + self.score_table[(self.r_full[j][k] + i) % 3][self.opp_history[-1]] for i in range(3)] for k in range(2)] for j in range(24)]
            self.p_freq_score[T%50] = [[[self.p_freq_score[(T+49)%50][j][k][i] + self.score_table[(self.p_freq[j][k] + i) % 3][self.opp_history[-1]] for i in range(3)] for k in range(2)] for j in range(2)]
            self.r_freq_score[T%50] = [[[self.r_freq_score[(T+49)%50][j][k][i] + self.score_table[(self.r_freq[j][k] + i) % 3][self.opp_history[-1]] for i in range(3)] for k in range(2)] for j in range(2)]
            self.s_len = [s + self.score_table[p][self.opp_history[-1]] for s,p in zip(self.s_len,self.p_len)]
    
    
        # update_history_hash()
        if not self.my_hist:
            self.my_history_hash = [[0],[0],[0],[0]]
            self.opp_history_hash = [[0],[0],[0],[0]]
        else:
            self.my_history_hash[0].append(self.my_history[-1])
            self.opp_history_hash[0].append(self.opp_history[-1])
            for i in range(1,4):
                self.my_history_hash[i].append(self.my_history_hash[i-1][-1] * 3 + self.my_history[-1])
                self.opp_history_hash[i].append(self.opp_history_hash[i-1][-1] * 3 + self.opp_history[-1])
    
    
        #make_predictions()
    
        for i in range(24):
            self.gear[i].append((3 + self.opp_history[-1] - self.p_full[i][2]) % 3)
            if T > 1:
                self.gear[i][T] += 3 * self.gear[i][T-1]
            self.gear[i][T] %= 9    # clearly there are 9 different gears, but original code only allocated 3 gear_freq's
                                    # code apparently worked, but got lucky with undefined behavior
                                    # I fixed by allocating gear_freq with length = 9
        if not self.my_hist:
            self.freq = [[0,0,0],[0,0,0]]
            value = [[0,0,0],[0,0,0]]
        else:
            self.freq[0][self.my_history[-1]] += 1
            self.freq[1][self.opp_history[-1]] += 1
            value = [[(1000 * (self.freq[i][2] - self.freq[i][1])) / float(T),
                      (1000 * (self.freq[i][0] - self.freq[i][2])) / float(T),
                      (1000 * (self.freq[i][1] - self.freq[i][0])) / float(T)] for i in range(2)]
        self.p_freq = [[self.wins_with[self.max_index(self.freq[i])], self.wins_with[self.max_index(value[i])]] for i in range(2)]
        self.r_freq = [[self.best_without[self.min_index(self.freq[i])], self.best_without[self.min_index(value[i])]] for i in range(2)]
    
        f = [[[[0,0,0] for k in range(4)] for j in range(2)] for i in range(3)]
        t = [[[0,0,0,0] for j in range(2)] for i in range(3)]
    
        m_len = [[0 for _ in range(T)] for i in range(3)]
    
        for i in range(T-1,0,-1):
            m_len[0][i] = 4
            for j in range(4):
                if self.my_history_hash[j][i] != self.my_history_hash[j][T]:
                    m_len[0][i] = j
                    break
            for j in range(4):
                if self.opp_history_hash[j][i] != self.opp_history_hash[j][T]:
                    m_len[1][i] = j
                    break
            for j in range(4):
                if self.my_history_hash[j][i] != self.my_history_hash[j][T] or self.opp_history_hash[j][i] != self.opp_history_hash[j][T]:
                    m_len[2][i] = j
                    break
    
        for i in range(T-1,0,-1):
            for j in range(3):
                for k in range(m_len[j][i]):
                    f[j][0][k][self.my_history[i+1]] += 1
                    f[j][1][k][self.opp_history[i+1]] += 1
                    t[j][0][k] += 1
                    t[j][1][k] += 1
    
                    if t[j][0][k] == 1:
                        self.p_full[j*8 + 0*4 + k][0] = self.wins_with[self.my_history[i+1]]
                    if t[j][1][k] == 1:
                        self.p_full[j*8 + 1*4 + k][0] = self.wins_with[self.opp_history[i+1]]
                    if t[j][0][k] == 3:
                        self.p_full[j*8 + 0*4 + k][1] = self.wins_with[self.max_index(f[j][0][k])]
                        self.r_full[j*8 + 0*4 + k][0] = self.best_without[self.min_index(f[j][0][k])]
                    if t[j][1][k] == 3:
                        self.p_full[j*8 + 1*4 + k][1] = self.wins_with[self.max_index(f[j][1][k])]
                        self.r_full[j*8 + 1*4 + k][0] = self.best_without[self.min_index(f[j][1][k])]
    
        for j in range(3):
            for k in range(4):
                self.p_full[j*8 + 0*4 + k][2] = self.wins_with[self.max_index(f[j][0][k])]
                self.r_full[j*8 + 0*4 + k][1] = self.best_without[self.min_index(f[j][0][k])]
    
                self.p_full[j*8 + 1*4 + k][2] = self.wins_with[self.max_index(f[j][1][k])]
                self.r_full[j*8 + 1*4 + k][1] = self.best_without[self.min_index(f[j][1][k])]
    
        for j in range(24):
            gear_freq = [0] * 9 # was [0,0,0] because original code incorrectly only allocated array length 3
    
            for i in range(T-1,0,-1):
                if self.gear[j][i] == self.gear[j][T]:
                    gear_freq[self.gear[j][i+1]] += 1
    
            #original source allocated to 9 positions of gear_freq array, but only allocated first three
            #also, only looked at first 3 to find the max_index
            #unclear whether to seek max index over all 9 gear_freq's or just first 3 (as original code)
            self.p_full[j][3] = (self.p_full[j][1] + self.max_index(gear_freq)) % 3
    
        # end make_predictions()
    
        self.p_len = [self.find_best_prediction(l, p_random) for l in self.lengths]
    
        return self.rps_to_num[self.rps_to_text[self.p_len[self.max_index(self.s_len)]]]
