# -*- coding: utf-8 -*-
"""
Created on Tue Nov 30 10:36:53 2021

@author: asus
"""
import matplotlib.pyplot as plt
import numpy as np

def ReadTrack(path):
    with open(path) as f:
        points = f.readlines()
        points = list(points)
    
    for i in range(len(points)):
        points[i] = points[i].replace('[', '')
        points[i] = points[i].replace(']', '')
        points[i] = points[i].split()
        for j in range(len(points[i])):
            num = int(points[i][j][-3 :])
            points[i][j] = float(points[i][j][0:-4]) * 10**num
            
    return points

def predict( x, P, A, Q ):
  x_minus = A.dot( x )
  P_minus = A.dot(P).dot(A.T) + Q
  return x_minus, P_minus

def correct( z, x_minus, P_minus, H, R ):
  di = np.linalg.inv( H.dot(P_minus).dot(H.T) + R)
  K = P_minus.dot(H.T).dot(di)
  x = x_minus + K.dot( z - H.dot(x_minus) )
  P = (np.eye(len(P_minus)) - K.dot( H )).dot(P_minus)
  return x, P

def Variance(arr):
    Sum = 0
    variance = []
    for i in range(len(arr)):
        Sum = Sum + arr[i]
        # mu
        mu = Sum/(i+1)
        #variance
        var = 0
        for j in range(i+1):
            var = var + (arr[j] - mu)**2
        variance.append(var/(i+1))
    return variance

def Tracking_Position():
    for num in range(1, 5):
        print('Track{}'.format(num))
        
        ''' read track '''
        path1 = 'inputs/track{}_true.txt'.format(num)
        path2 = 'inputs/track{}_observe.txt'.format(num)
        true = ReadTrack(path1) # True track
        observe = ReadTrack(path2) # Observe track
        true = np.array(true)
        observe = np.array(observe)
        
        ''' Kalman Filter '''
        x = true[0]
        A = np.array([[1, 0], [0, 1]])
        Q = np.array([[0.5, 0], [0, 0.5]])
        H = np.array([[1.0, 0], [0, 1.0]])
        R = np.array([[1, 0], [0, 1]])
        P = np.array([[1, 0], [0, 1]])
        N_points = len(true)
        
        x_kalman = []
        y_kalman = []
        
        for i in range(N_points):
            x_minus, P_minus = predict(x, P, A, Q)
            x, P = correct(observe[i], x_minus, P_minus, H, R)
            x_kalman.append(x[0])
            y_kalman.append(x[1])
        
        ''' calculate error '''
        error = []
        for i in range(N_points):
            e = 0
            for j in range(i+1):
                e = e + np.sqrt((x_kalman[j]-true[j][0])**2 + (y_kalman[j]-true[j][1])**2)
            error.append(e/(i+1))
        
        ''' calculate variance of error '''
        var = Variance(error)
        
        ''' plot '''
        x_true = []
        y_true = []
        for i in range(len(true)):
            x_true.append(true[i][0])
            y_true.append(true[i][1])
        
        x_observe = []
        y_observe = []
        for i in range(len(observe)):
            x_observe.append(observe[i][0])
            y_observe.append(observe[i][1])
        
        # track robot
        plt.plot(x_true, y_true, label='True')
        plt.plot(x_observe, y_observe, label='Measured')
        plt.plot(x_kalman, y_kalman, label='Filtered')
        plt.title('track{}: tracked robot'.format(num))
        plt.xlabel("x") 
        plt.ylabel("y") 
        plt.legend()
        plt.savefig('outputs/Position/track{}/track_position{}.png'.format(num, num))
        # plt.show()
        
        fig, (axs3, axs4) = plt.subplots(1, 2)
        dt = 0.01
        t = np.linspace(0, N_points-1, num=N_points)
        t = t* dt
        # error
        axs3.plot(t, error)
        axs3.set_title('track{}: error vs t'.format(num))
        
        # variance of error
        axs4.plot(t, var)
        axs4.set_title('track{}: variance vs t'.format(num))
        fig.savefig("outputs/Position/track{}/track{}_error_and_variance.png".format(num, num))
        plt.show()





