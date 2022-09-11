# -*- coding: utf-8 -*-
"""
Created on Tue Nov 30 19:34:16 2021

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


''' Kalman Filter tracking acceleration '''
def Tracking_Acceleration():
    for num in range(1, 5):
        print('Track{}'.format(num))
        
        ''' read track '''
        path1 = 'inputs/track{}_true.txt'.format(num)
        path2 = 'inputs/track{}_observe.txt'.format(num)
        # path1 = 'mytracks/track2_true.txt'
        # path2 = 'mytracks/track2_observe.txt'
        true = ReadTrack(path1) # True track
        observe = ReadTrack(path2) # Observe track
        true = np.array(true)
        observe = np.array(observe)
        
        ''' calculate velocity '''
        dt = 0.01
        N_points = len(true)
        true_v = [[0, 0]] # true track with velocity
        # observe_v = [[0, 0]] # observe track with velocity
        for i in range(1, N_points):
            true_v.append([(true[i][0]-true[i-1][0])/dt, (true[i][1]-true[i-1][1])/dt])
            # observe_v.append([(observe[i][0]-observe[i-1][0])/dt, (observe[i][1]-observe[i-1][1])/dt])
        
        ''' calculate acceleration '''
        true_a = [[0, 0]]
        # observe_a = [[0, 0]]
        for i in range(1, N_points):
            true_a.append([(true_v[i][0]-true_v[i-1][0])/dt, (true_v[i][1]-true_v[i-1][1])/dt])
            # observe_a.append([(observe_v[i][0]-observe_v[i-1][0])/dt, (observe_v[i][1]-observe_v[i-1][1])/dt])
        
        ''' Kalman Filter '''
        x = [true[2][0], true[2][1], true_v[2][0], true_v[2][1], true_a[2][0], true_a[2][1]]
        
        A = np.array([[1, 0, dt, 0, 0, 0], [0, 1, 0, dt, 0, 0], [0, 0, 1, 0, dt, 0]
                      , [0, 0, 0, 1, 0, dt], [0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 1]])
        
        Q = np.array([[0.1, 0, 0, 0, 0, 0], [0, 0.1, 0, 0, 0, 0], [0, 0, 20, 0, 0, 0]
                      , [0, 0, 0, 20, 0, 0], [0, 0, 0, 0, 1000, 0], [0, 0, 0, 0, 0, 1000]])
        
        H = np.array([[1.0, 0, 0, 0, 0, 0], [0, 1.0, 0, 0, 0, 0]])
        
        R = np.array([[100, 0], [0, 100]])
        
        P = np.array([[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0],[0, 0, 1, 0, 0, 0],
                      [0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 1]])
        
        x_kalman_a = []
        y_kalman_a = []
        
        for i in range(2, N_points):
            x_minus, P_minus = predict(x, P, A, Q)
            x, P = correct([observe[i][0], observe[i][1]], x_minus, P_minus, H, R)
            x_kalman_a.append(x[4])
            y_kalman_a.append(x[5])
        
        ''' calculate error '''
        error = []
        for i in range(2, N_points):
            e = 0
            for j in range(2, i+1):
                e = e + np.sqrt((x_kalman_a[j-2]-true_a[j-2][0])**2 + (y_kalman_a[j-2]-true_a[j-2][1])**2)
            error.append(e/(i-2+1))
        
        ''' calculate variance of error '''
        var = Variance(error)
        
        ''' plot '''
        x_true_a = []
        y_true_a = []
        for i in range(2, len(true_a)):
            x_true_a.append(true_a[i][0])
            y_true_a.append(true_a[i][1])
        
        dt = 0.01
        t = np.linspace(0, N_points-1, num=N_points)
        t = t* dt
        fig, (axs1, axs2) = plt.subplots(1, 2)
        # x dot
        # axs1.plot(t, x_observe_a, linewidth=1, label='Measured')
        axs1.plot(t[2:], x_kalman_a, linewidth=2, label='Filtered')
        axs1.plot(t[2:], x_true_a, linewidth=2, label='True')
        axs1.set_title('track{}: x double-dot v.s. t'.format(num))
        axs1.legend()
        
        # y dot
        # axs2.plot(t, y_observe_a, linewidth=1, label='Measured')
        axs2.plot(t[2:], y_kalman_a, linewidth=2, label='Filtered')
        axs2.plot(t[2:], y_true_a, linewidth=2, label='True')
        axs2.set_title('track{}: y double-dot v.s. t'.format(num))
        axs2.legend()
        fig.savefig('outputs/Acceleration/track{}/track_acceleration{}.png'.format(num, num))
        
        fig, (axs3, axs4) = plt.subplots(1, 2)
        dt = 0.01
        t = np.linspace(0, N_points-1, num=N_points)
        t = t* dt
        # error
        axs3.plot(t[2:], error)
        axs3.set_title('track{}: error vs t'.format(num))
        
        # variance of error
        axs4.plot(t[2:], var)
        axs4.set_title('track{}: variance vs t'.format(num))
        fig.savefig('outputs/Acceleration/track{}/track{}_error_and_variance.png'.format(num, num))
        plt.show()
    