from matplotlib import pyplot as plt
import os   
import numpy as np
import sys

class DataPlotter(object):
    def __init__(self):
        self.thedict = {}
        self.clear_data_list()

        self.cut_length = 50

        self.start_ind = 0
        self.end_ind = 1000
        self.step = 100
        self.init = ''
        self.delta = ''

        return

    def clear_data_list(self):
        self.time_list = []
        self.u_list = []
        self.y_list = []
        self.y_out_list = []

        return

    def extract_file(self,file):
        if os.path.exists(file):

            with open(file,'r') as f:
                count = 0
                for line in f:
                    if count < 1:
                        count = count + 1
                        continue
                    if count < self.start_ind:
                        count += 1
                        continue
                    if count > self.end_ind:
                        break
                    line_sp = line.split(',')

                    timestamp = float(line_sp[0])
                    u = float(line_sp[1])
                    y = float(line_sp[2])
                    y_noise = float(line_sp[3])
                    self.time_list.append(timestamp)
                    self.u_list.append(u)
                    self.y_list.append(y)
                    self.y_out_list.append(y_noise)

                    count += 1
        return
    
    def data_plot(self):

        plt.figure()
        plt.subplot(2,1,1)
        plt.plot(self.u_list,self.y_list,label='u-y')
        plt.plot(self.u_list,self.y_out_list,label='u-y_noise')
        plt.grid()
        plt.legend()

        plt.subplot(2,1,2)
        plt.plot(self.u_list,label='u')
        plt.plot(self.y_list,label='y')
        plt.plot(self.y_out_list,label='y with noise')
        plt.grid()
        plt.legend()
    

def main(root):
    if len(sys.argv) < 2:
        print ("ERROR: Too few arguments, please add log file")

    file_name = sys.argv[1]

    plotter = DataPlotter()
    plotter.extract_file(file_name)
    plotter.data_plot()

    plt.show()


if __name__ == '__main__':
    main(".")