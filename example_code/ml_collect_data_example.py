
"""
The template of the main script of the machine learning process
"""
import pickle
import os
import time
import numpy as np

class MLPlay:
    def __init__(self, *args, **kwargs):
        """
        Constructor
        """
        self.ball_served = False
        self.previous_ball = (0, 0)     
        self.pred = 100                 # Prediction of board x axis location
        self.platform_y = 400           # Position of board y axis
        self.ball_speed_y = 7           # Ball vertical speed
        self.window_width = 200         # Width of the game window 
        
        self._ml_names = ["1P"]
        game_progress = {
            "record_format_version": 2
        }
        for name in self._ml_names:
            game_progress[name] = {
                "scene_info": [],
                "command": []
            }
        self._game_progress = game_progress

    def update(self, scene_info, *args, **kwargs):
        
        if (scene_info["status"] == "GAME_OVER" or
                scene_info["status"] == "GAME_PASS"):
            return "RESET"

        if not self.ball_served:            
            self.ball_served = True
            self.previous_ball = scene_info["ball"]
            if np.random.random() < 0.5:
                command = "SERVE_TO_RIGHT"      # You can change the direction to serve ball
            else:
                command = "SERVE_TO_LEFT"
        else:
            """
            You can implement your code here to pass game automatically.
            """

            speed_x = scene_info["ball"][0] - self.previous_ball[0]
            speed_y = scene_info["ball"][1] - self.previous_ball[1]
            pred = scene_info["ball"][0] + ((scene_info["platform"][1] - scene_info["ball"][1]) // speed_y) * speed_x


            if scene_info["platform"][0] < pred:
                command = "MOVE_RIGHT"
            else:
                command = "MOVE_LEFT"

        self.previous_ball = scene_info["ball"]

        # Pass scene_info and command to generate data
        self.record(scene_info, command)
        return command

    def reset(self):
        """
        Reset the status
        """
        self.flush_to_file()
        self.ball_served = False

    def record(self, scene_info_dict: dict, cmd_dict: dict):
        """
        Record the scene information and the command
        The received scene information will be stored in a list.
        """
        for name in self._ml_names:
            self._game_progress[name]["scene_info"].append(scene_info_dict)
            self._game_progress[name]["command"].append(cmd_dict)

    def flush_to_file(self):
        """
        Flush the stored objects to the file
        """
        filename = time.strftime("%Y-%m-%d_%H-%M-%S") + ".pickle"
        if not os.path.exists(os.path.dirname(__file__) + "/log"):
            os.makedirs(os.path.dirname(__file__) + "/log")
        filepath = os.path.join(os.path.dirname(__file__),"./log/" + filename)
        # Write pickle file
        with open(filepath, "wb") as f:
            pickle.dump(self._game_progress, f)

        # Clear list
        for name in self._ml_names:
            target_slot = self._game_progress[name]
            target_slot["scene_info"].clear()
            target_slot["command"].clear()