'''
Created on Jan. 13, 2021

@author: zollen
@url: http://www.rpscontest.com/entry/885001
'''

import random
import rps_kaggle.lib.rps_lib as rps


class Iocaine(rps.BaseAgent):
    
    def __init__(self, states = 3, window = 5, ahead = 0, counter = None, num_predictor = 27):
        super().__init__(states, window, ahead, counter)
        self.SYMBOLS = {0: 'R', 1: 'P', 2: 'S'}
        self.RSYMBOLS = { 'R': 0, 'P': 1, 'S': 2 }
        self.len_rfind = [20]
        self.limit = [10,20,60]
        self.beat = { "R":"P" , "P":"S", "S":"R"}
        self.not_lose = { "R":"PPR" , "P":"SSP" , "S":"RRS" } #50-50 chance
        self.my_his   =""
        self.your_his =""
        self.both_his =""
        self.num_predictor = num_predictor
        self.list_predictor = [""]*self.num_predictor
        self.score_predictor = [0]*self.num_predictor
        self.length = 0
        self.temp1 = {      "PP":"1" , "PR":"2" , "PS":"3",
                            "RP":"4" , "RR":"5", "RS":"6",
                            "SP":"7" , "SR":"8", "SS":"9"}
        self.temp2 = {      "1":"PP","2":"PR","3":"PS",
                            "4":"RP","5":"RR","6":"RS",
                            "7":"SP","8":"SR","9":"SS"} 
        self.who_win = {    "PP": 0, "PR":1 , "PS":-1,
                            "RP": -1,"RR":0, "RS":1,
                            "SP": 1, "SR":-1, "SS":0}
        self.output = None
        self.input = None
        self.last = None
        self.predictors = None
    
    def deposit(self, token):
        self.output = self.SYMBOLS[token]
        self.last = token
        super().deposit(token)
        
    def add(self, token):
        self.input = self.SYMBOLS[token]
        super().add(token)
        
    def __str__(self):
        return "Iocaine2(" + str(self.num_predictor) + ")"
        
    def random(self):
        return self.SYMBOLS[random.randint(0, 2)]
        
    def decide(self):
        
        if self.input == None:
            self.output = self.random()
            self.predictors = [self.output] * self.num_predictor
            self.last = self.submit(self.RSYMBOLS[self.output])
            return self.last
        
        if len(self.list_predictor[0])<5:
            front =0
        else:
            front =1
        
        for i in range (self.num_predictor):
            if self.predictors[i]==self.input:
                result ="1"
            else:
                result ="0"
            self.list_predictor[i] = self.list_predictor[i][front:5]+result #only 5 rounds before

        #history matching 1-6
        self.my_his += self.output
        self.your_his += self.input
        self.both_his += self.temp1[self.input+self.output]
        self.length +=1
        
        for i in range(1):
            len_size = min(self.length,self.len_rfind[i])
            j=len_size
            #both_his
            # extract a maximun length token that contains in both_his[0, self.legnth - 1]
            while j>=1 and not self.both_his[self.length-j:self.length] in self.both_his[0:self.length-1]:
                j-=1
            if j>=1:
                # find the starting position of the matched token
                k = self.both_his.rfind(self.both_his[self.length-j:self.length],0,self.length-1)
                # extract the next action just after the matched token
                self.predictors[0+6*i] = self.your_his[j+k]
                self.predictors[1+6*i] = self.beat[self.my_his[j+k]]
            else:
                self.predictors[0+6*i] = random.choice("RPS")
                self.predictors[1+6*i] = random.choice("RPS")
                
            j=len_size
            #your_his
            # extract a maximun length token that contains in your_his[0, self.legnth - 1]
            while j>=1 and not self.your_his[self.length-j:self.length] in self.your_his[0:self.length-1]:
                j-=1
            if j>=1:
                # find the starting position of the matched token
                k = self.your_his.rfind(self.your_his[self.length-j:self.length],0,self.length-1)
                # extract the next action just after the matched token
                self.predictors[2+6*i] = self.your_his[j+k]
                self.predictors[3+6*i] = self.beat[self.my_his[j+k]]
            else:
                self.predictors[2+6*i] = random.choice("RPS")
                self.predictors[3+6*i] = random.choice("RPS")
                
            j=len_size
            #my_his
            # extract a maximun length token that contains in my_his[0, self.legnth - 1]
            while j>=1 and not self.my_his[self.length-j:self.length] in self.my_his[0:self.length-1]:
                j-=1
            if j>=1:
                # find the starting position of the matched token
                k = self.my_his.rfind(self.my_his[self.length-j:self.length],0,self.length-1)
                # extract the next action just after the matched token
                self.predictors[4+6*i] = self.your_his[j+k]
                self.predictors[5+6*i] = self.beat[self.my_his[j+k]]
            else:
                self.predictors[4+6*i] = random.choice("RPS")
                self.predictors[5+6*i] = random.choice("RPS")

        for i in range(3):
            temp =""
            # get the encoded value of the last my move and opponent move
            search = self.temp1[(self.output+self.input)] #last round
            # From 2 -> min(10, self.length)
            # From 2 -> min(20, self.length)
            # From 2 -> min(60, self.length)
            #     Search self.both_his backward 
            #     Get the next encoded action(s) after the matched token(s) 
            for start in range(2, min(self.limit[i],self.length) ):
                if search == self.both_his[self.length-start]:
                    temp+=self.both_his[self.length-start+1]
            if(temp==""):
                self.predictors[6+i] = random.choice("RPS")
            else:
                collectR = {"P":0,"R":0,"S":0} #take win/lose from opponent into account
                # Decode each likely action in temp and update the scorebaord collectR
                for sdf in temp:
                    next_move = self.temp2[sdf]
                    if(self.who_win[next_move]==-1):
                        collectR[self.temp2[sdf][1]]+=3
                    elif(self.who_win[next_move]==0):
                        collectR[self.temp2[sdf][1]]+=1
                    elif(self.who_win[next_move]==1):
                        collectR[self.beat[self.temp2[sdf][0]]]+=1
                max1 = -1
                p1 =""
                # Find the likely action with the highest count from the scoreboard
                for key in collectR:
                    if(collectR[key]>max1):
                        max1 = collectR[key]
                        p1 += key
                self.predictors[6+i] = random.choice(p1)
    
        #rotate 9-27:
        for i in range(9,27):
            self.predictors[i] = self.beat[self.beat[self.predictors[i-9]]]
        
        #choose a predictor
        len_his = len(self.list_predictor[0])
        for i in range(self.num_predictor):
            summ = 0
            for j in range(len_his):
                if self.list_predictor[i][j]=="1":
                    summ += (j+1)*(j+1)
                else:
                    summ -= (j+1)*(j+1)
            self.score_predictor[i] = summ
        max_score = max(self.score_predictor)
   
        if max_score>0:
            self.predict = self.predictors[self.score_predictor.index(max_score)]
        else:
            self.predict = random.choice(self.your_his)
        self.output = random.choice(self.not_lose[self.predict])
        self.last = self.RSYMBOLS[self.output]
        return self.submit(self.last)