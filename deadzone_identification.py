from threading import Thread
import numpy as np
import math
import os
import traceback

import rs_method
import matplotlib.pyplot as plt
import control as ctrl

import filter

# deadzone + first_order_element
# deadzone: u(k) = if v > d_r v(k) - d_r else 0
# first_order_element: G(s) =  a1 / (s+a0)

class ModelIdentification(object):

    def __init__(self):
        # data
        self.clear_data_list()
        self.T = 0.01
        self.test_info = ''
        self.start_ind = 0
        self.end_ind = -1
        # deazone
        self.d_r_star = 1.2

        self.dict_a0 = {}
        return

    def clear_data_list(self):
        self.time_list = []
        self.u_list = []
        self.y_list = []
        self.y_out_list = []
        self.y_filter_list = []
        self.filter = filter.Filters(10)

        return

    def extract_file(self,file):
        if os.path.exists(file):
            with open(file,'r') as f:
                count = 0
                time_start = 0
                for line in f:
                    if count < 1:
                        count = count + 1
                        continue

                    line_sp = line.split(',')
                    if count == 1:
                        time_start = float(line_sp[0])
                        count = count + 1

                    timestamp = float(line_sp[0])
                    u = float(line_sp[1])
                    y = float(line_sp[2])
                    y_noise = float(line_sp[3])
                    self.time_list.append(timestamp)
                    self.u_list.append(u)
                    self.y_list.append(y)
                    self.y_out_list.append(y_noise)
            # filter
            for y_out in self.y_out_list:
                y_filtered = self.filter.mean_filter(y_out)
                self.y_filter_list.append(y_filtered)
        return

    def get_x_y_list(self,use_filter=False):
        if use_filter:
            y = self.y_filter_list[self.start_ind:self.end_ind]
        else:
            y = self.y_out_list[self.start_ind:self.end_ind]
        u = self.u_list[self.start_ind:self.end_ind]
        return u,y

    # ARMA model: difference
    # y(n-1) = [y(n),u(n)] * [a0*T+1,a1*T]'
    # y(n) = a*y(n-1) + b*v(n) - b*d_r if v(n) >= d_r_star
    #      = a*y(n-1) if v(n) < d_r_star
    # where, a=1/(a0*T+1), b=a1*T/(a0*T+1)
    # PSI = [y(n-1),v(n),1], THETA = [a,b,-b*c_r]
    # unknown parameters: a0,a1,d_r
    # assumption: d_r_star > d_r

    def ARMA_model(self,x_list,y_list):
        Y = np.array([y_list[1:]])
        x1 = y_list[:-1]
        x2 = x_list[1:]
        x3 = [1 for i in x2]
        X = np.array([x1,x2,x3])

        # cutting: v(k)<d_r_star
        YY = []
        XX = []
        for Yi,Xi in zip(Y.T,X.T):
            if Xi[1] >= self.d_r_star:
                YY.append(Yi)
                XX.append(Xi)
        YY = np.array(YY)
        XX = np.array(XX)
        data = np.concatenate((YY,XX),axis=1)
        return data

    def back_differential(self,u_list,y_list):
        data = self.ARMA_model(u_list,y_list)
        featureNum = 3
        Theta,theta_array = self.ls_method(data,featureNum)
        print("Theta: ",Theta)
        # solve a0,a1
        res = self.solve_a_diff(Theta)

        return res,theta_array

    def solve_a_diff(self,Theta):
        a = Theta[0][0]
        b = Theta[1][0]
        b_d = Theta[2][0]

        a0 = (1.0/a-1)/self.T
        a1 = b*(a0*self.T+1)/self.T
        d_r = -b_d/b

        return [a0,a1,d_r]

    def ls_method(self,data,featureNum=2):
        initialTheta = 0.5 * np.ones((featureNum, 1))
        Theta,theta_array = rs_method.RLS_Fun(data, initialTheta, featureNum)
        return Theta,theta_array

    def show_theta(self,theta_array):
        plt.figure()
        lengend_list = []
        for i in np.arange(len(theta_array)):
            plt.plot(theta_array[i])
            lengend_list.append("theta_"+str(i))
        plt.legend(lengend_list)
        plt.grid()
        return

    def show_fitting(self):
        a0 = self.res[0]
        a1 = self.res[1]
        d_r = self.res[2]
        s = ctrl.tf('s')
        sys = a1 / (s + a0)

        # signals
        T = [i*self.T for i in np.arange(len(self.time_list))]
        X0 = 0
        U = []
        for u in self.u_list:
            if u >= d_r:
                U.append(u-d_r)
            else:
                U.append(0.0)

        t, response = ctrl.forced_response(sys, T, U, X0)

        plt.figure()
        plt.plot(self.time_list,self.u_list,label='u')
        plt.plot(self.time_list,self.y_list,label='y')
        plt.plot(self.time_list,self.y_out_list,label='y_noise')
        plt.plot(self.time_list,self.y_filter_list,label='y_filter')
        plt.plot(t,response,label='y_fitting')

        plt.legend()
        plt.grid()

        return

    def data_show(self,theta_array):
        self.show_theta(theta_array)
        self.show_fitting()
        return

    def run(self,file):
        file_split = file.split('_')
        self.test_info = file_split
        self.extract_file('./data/'+file)
        u_list,y_list = self.get_x_y_list(use_filter=True)
        self.res,self.theta_array = self.back_differential(u_list,y_list)

        print("res: ",self.res)
        # print(theta_array)
        # Show
        self.data_show(self.theta_array)
        return


def main():
    identifier = ModelIdentification()
    identifier.run('test_data.txt')
    
    plt.show()

if __name__ == '__main__':
    main()

    

    