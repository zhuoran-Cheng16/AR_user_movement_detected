# -*- encoding: utf-8 -*-
"""
Copyright (c) 2019 - present AppSeed.us
"""

from flask_migrate import Migrate
from os import environ
from sys import exit
from decouple import config
import logging

from config import config_dict
from app import create_app, db
from Inference import ExerciseCapture
from time import time
from time import sleep

from Utils import Utils

import threading
import sys
import cv2

from warnings import filterwarnings
filterwarnings('ignore')
# WARNING: Don't run with debug turned on in production!
DEBUG = config('DEBUG', default=True, cast=bool)

# The configuration
get_config_mode = 'Debug' if DEBUG else 'Production'
class Workout:
    def __init__(self):

        self.ex = ExerciseCapture()
        self.currentExercise = 'None'
        self.training_stats = {}
        self.IM_SIZE = (128, 128)
        self.utils = Utils()


        self.isStarted = False
        self.isFinished = False
        self.isRest = False
        self.isTabata = False
        self.playSound = False
        self.playSoundFinish = False
        self.thresh = 0
        self.timeToStart = 0
def runTraining(self, training_program, models, tabata, restTimes):
        '''
        Performs move counting and stats collection for each exercise while training. Can perform tabata or just simple move counting
        :param training_program: The program on which training is performed
        :param models: models for movement counting for each exercise
        :param tabata: If the training should be tabata or not
        :param restTime: time to rest between each exercise
        :return: Collected stats for each training
        '''
        self.isTabata = tabata
        self.isFinished = False
        t = threading.currentThread()
        timeStart = time()
        timeEx = 0
        totalMoves = 0
        cap = cv2.VideoCapture(0)
        for index, (key, value) in enumerate(training_program.items()):
            self.currentExercise = key
            model = models[key]
            if not tabata:
                print(f'Performing {value} {key}')
            else:
                print(f'Performing {key} for {value} seconds')

            self.thresh = value
            self.ex = ExerciseCapture(model, fromStream=True, timeWise=tabata, thresh=value, name=key)
            self.isStarted = True
            self.playSound = True
            moves, totalTime = self.ex.runPipeline(cap)


            timeEx += totalTime
            totalMoves += moves

            self.training_stats[key+f'_{str(index)}'] = {'moves': moves, 'time': totalTime}
            print(f'Exercise {key} finishes; Moves {moves} Time {totalTime}')

            print(f'Rest for {restTimes[index]} seconds')

            timeStarted = time()
            self.playSoundFinish = True
            while time() - timeStarted < restTimes[index]:
                self.isRest = True
                self.timeToStart = restTimes[index] - (time() - timeStarted)
                self.ex.origFrame, _, _= self.utils.readFrame(cap, self.IM_SIZE)
                if not t.do_run:
                    self.isStarted = False
                    break
            if not t.do_run:
                self.isStarted = False
                break
            self.isRest = False

        cap.release()
        cv2.destroyAllWindows()

        self.isStarted = False
        self.ex.origFrame = cv2.imread('blank.png')
        trainingTime = time() - timeStart
        restingTime = trainingTime - timeEx
        print(f'Training has taken {trainingTime}. You have been resting for {restingTime} and performing for {timeEx} seconds \n Total moves {totalMoves}')

        self.training_stats['totalTime'] = trainingTime
        self.training_stats['restTime'] = restingTime
        self.training_stats['exerciseTime'] = timeEx
        self.training_stats['totalMoves'] = int(totalMoves)

        self.isFinished = True

        #return self.training_stats
try:
    
    # Load the configuration using the default values 
    app_config = config_dict[get_config_mode.capitalize()]

except KeyError:
    exit('Error: Invalid <config_mode>. Expected values [Debug, Production] ')

app = create_app( app_config ) 
Migrate(app, db)

if DEBUG:
    app.logger.info('DEBUG       = ' + str(DEBUG)      )
    app.logger.info('Environment = ' + get_config_mode )
    app.logger.info('DBMS        = ' + app_config.SQLALCHEMY_DATABASE_URI )

if __name__ == "__main__":
    app.run()
