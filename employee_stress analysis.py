import sys
from datetime import datetime
from kivy.clock import Clock
import time
import matplotlib.pyplot as plt
from typing import Dict, List
import cv2
import numpy as np
from emotion_analyzer.emotion_detector import EmotionDetector
from emotion_analyzer.logger import LoggerFactory
from emotion_analyzer.media_utils import (
    annotate_emotion_stats,
    annotate_warning,
    convert_to_rgb,
    draw_bounding_box_annotation,
    draw_emoji,
)
from emotion_analyzer.validators import path_exists
import cv2
from kivy.uix.image import Image
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.label import Label
from kivy.app import App
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.graphics import Color, Rectangle
from kivy.core.window import Window
from kivy.uix.button import Button
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.textinput import TextInput
import os
from plotter import pol

closer=1
sl=[]

class EmotionAnalysisVideo:
    emoji_foldername = "emojis"
    def __init__(
        self,
        face_detector: str = "dlib",
        model_loc: str = "models",
        face_detection_threshold: float = 0.8,
        emoji_loc: str = "data",
    ) -> None:
        self.emoji_path = os.path.join(emoji_loc, EmotionAnalysisVideo.emoji_foldername)
        self.emojis = self.load_emojis(emoji_path=self.emoji_path)
        self.emotion_detector = EmotionDetector(
            model_loc=model_loc,
            face_detection_threshold=face_detection_threshold,
            face_detector=face_detector,
        )
    def emotion_analysis_video(
        self,
        detection_interval: int = 1,
        save_output: bool = False,
        preview: bool = True,
        output_path: str = "data/output.mp4",
        resize_scale: float = 0.5,
    ) -> None:
        cap= None
        detection_interval=1
        cap = cv2.VideoCapture(0)
        frame_num = 1
        t1 = time.time()         
        emotions = None
        closer=1
        while(time.time() -t1 < 30):
            status, frame = cap.read()
            if not status:
                break
            frame = cv2.flip(frame, 2)
            if frame_num % detection_interval == 0:
                # Scale down the image to increase model # inference time.
                smaller_frame = convert_to_rgb(cv2.resize(frame, (0, 0), fx=resize_scale, fy=resize_scale))
                # Detect emotion
                emotions = self.emotion_detector.detect_emotion(smaller_frame)
                # Annotate the current frame with emotion detection data
                frame = self.annotate_emotion_data(emotions, frame, resize_scale)
                cv2.imshow("Preview", cv2.resize(frame, (700,500)))

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                closer=0
                print("Value of ex   --- ",closer)
                
                print("----------> executed --------")
                break
                
            frame_num += 1
                       

        cv2.destroyAllWindows()
        cap.release()
        return closer


    def load_emojis(self, emoji_path: str = "data//emoji") -> List:
        emojis = {}

        # list of given emotions
        EMOTIONS = [
            "Angry",
            "Disgusted",
            "Fearful",
            "Happy",
            "Sad",
            "Surprised",
            "Neutral",
        ]

        # store the emoji coreesponding to different emotions
        for _, emotion in enumerate(EMOTIONS):
            emoji_path = os.path.join(self.emoji_path, emotion.lower() + ".png")
            emojis[emotion] = cv2.imread(emoji_path, -1)

        print("Finished loading emojis...")

        return emojis

    def annotate_emotion_data(
        self, emotion_data: List[Dict], image, resize_scale: float
    ) -> None:

        # draw bounding boxes for each detected person
        for data in emotion_data:
           
            image = draw_bounding_box_annotation(
                image, data["emotion"], int(1 / resize_scale) * np.array(data["bbox"])
            )
            sl.append(data["emotion"])
        
            

        # If there are more than one person in frame, the emoji can be shown for
        # only one, so show a warning. In case of multiple people the stats are shown
        # for just one person
        WARNING_TEXT = "Warning ! More than one person detected !"

        if len(emotion_data) > 1:
            image = annotate_warning(WARNING_TEXT, image)

        if len(emotion_data) > 0:
            print("vivek testing ",emotion_data[0])
            # draw emotion confidence stats
            image = annotate_emotion_stats(emotion_data[0]["confidence_scores"], image)
            # draw the emoji corresponding to the emotion
            image = draw_emoji(self.emojis[emotion_data[0]["emotion"]], image)
            
        return image

def viv():
    ft=open('time.txt','a')
    fs=open('stress.txt','a')
    f=open('Final report.txt','a')
    current_datetime = datetime.now()
    formatted_datetime = current_datetime.strftime("%d-%m-%Y")
    f.write('\n\tDATE  --  '+formatted_datetime+'\nSTRESS LEVEL\t\tTIME')
    emotion_recognizer = EmotionAnalysisVideo(
                        face_detector="dlib",
                        model_loc="models",
                        face_detection_threshold=0.0,
                    )
    c=1
    while(c==1):
        c=emotion_recognizer.emotion_analysis_video(
        detection_interval=1,
        save_output=False,
        preview=True,
        output_path="data/output.mp4",
        resize_scale=0.5,
         )
        stress=0
        for w in sl:
            if(w=='Angry'):
                stress=stress+2
            elif(w=='Neutral'):
                stress=stress-0.4
            elif(w=='Sad'):
                stress=stress+2
            elif(w=='Happy'):
                stress=stress-2
            elif(w=='Disgust'):
                stress=stress+2
            elif(w=='Surprised'):
                stress=stress-2
       
        if(stress<=0):
            stress=0
        current_datetime = datetime.now()
        formatted_datetime = current_datetime.strftime("%H:%M:%S")
        stress=round(stress,1)
        f.write('\n'+formatted_datetime+'\t--->\t'+str(stress))
        fs.write(str(stress)+'\n')
        ft.write(formatted_datetime+'\n')
        sl.clear()
        print('----------> Stress ===',stress)
        print("stress list------->",sl)  
        if(c==1):
            time.sleep(10)  


#-------------------------gui------------gui--------------------giu---------------------------------------------

class Introduction(Screen):
    def __init__(self, **kwargs):
        super(Introduction, self).__init__(**kwargs)
        layout = FloatLayout()
        with layout.canvas.before:
            Color(127, 0, 255, 0.4)  # RGBA values (red, green, blue, alpha)
            self.rect = Rectangle(size=layout.size, pos=layout.pos)
        layout.bind(size=self._update_rect, pos=self._update_rect)
        image = Image(source='emp.png',size=(1400,1000))
        tlabel = Label(text='Welcome', pos_hint={'x': 0, 'y': .1}, font_size=120)
        alabel = Label(text='AI Employee Stress Monitoring System', pos_hint={'top': .9, 'bottom': .1}, font_size=45, )
        button = Button(text="Login", size=(20, 50), size_hint=(0.1, 0.1), pos_hint={'top': .25, 'right': .55},
                        font_size=20, on_press=self.go_to_login)
        nlabel = Label(text='build  @Vivek Rajeev  @Anandu Ramesh @Delin Benny @Mohit Prakash', pos_hint={'x': 0,'y':-0.46},font_size=14)
        
        layout.add_widget(image)
        layout.add_widget(nlabel)
        layout.add_widget(button)
        layout.add_widget(tlabel)
        layout.add_widget(alabel)
        self.add_widget(layout)

    def _update_rect(self, instance, value):
        # Update the size and position of the rectangle
        self.rect.size = instance.size
        self.rect.pos = instance.pos

    def go_to_login(self, instance):
        self.manager.current = 'login'

class LoginPage(Screen):
    def __init__(self, **kwargs):
        super(LoginPage, self).__init__(**kwargs)
        alayout = FloatLayout()
        with alayout.canvas.before:
            Color(127, 0, 255, 0.4)  # RGBA values (red, green, blue, alpha)
            self.rect = Rectangle(size=alayout.size, pos=alayout.pos)
        alayout.bind(size=self._update_rect, pos=self._update_rect)
        layout = BoxLayout(orientation='vertical', size_hint=(0.4, 0.4), pos_hint={'x': .3, 'y': .4})
        username_label = Label(text='Username:')
        self.username_input = TextInput(multiline=False)
        password_label = Label(text='Password:')
        self.password_input = TextInput(password=True, multiline=False)
        login_button = Button(text='Login', size_hint=(.30, .15), pos_hint={'x': .35, 'y': .1})
        login_button.bind(on_press=self.login)

        layout.add_widget(username_label)
        layout.add_widget(self.username_input)
        layout.add_widget(password_label)
        layout.add_widget(self.password_input)
        alayout.add_widget(login_button)

        alayout.add_widget(layout)
        self.add_widget(alayout)

    
    def login(self, instance):
        # Get username and password input values
        username = self.username_input.text
        password = self.password_input.text
        # Validate username and password (dummy validation for demonstration)
        if username == u1 and password == u2:
            print('Login successful!')
            self.employeepage()
        elif username== a1 and password == a2:
            self.adminpage()

    def _update_rect(self, instance, value):
        # Update the size and position of the rectangle
        self.rect.size = instance.size
    
    def employeepage(self):
        self.manager.current = 'a'
    def adminpage(self):
        self.manager.current = 'ad'

class Employee(Screen):
    def __init__(self, **kwargs):
        super(Employee, self).__init__(**kwargs)
        layout = FloatLayout()
        with layout.canvas.before:
            Color(127, 0, 255, 0.4)  # RGBA values (red, green, blue, alpha)
            self.rect = Rectangle(size=layout.size, pos=layout.pos)
        layout.bind(size=self._update_rect, pos=self._update_rect)
        tlabel = Label(text='Employee Page ', pos_hint={'x': 0, 'y': .1}, font_size=60)
        alabel = Label(text='Click on start to begin Monitoring', pos_hint={'top': .9, 'bottom': .1}, font_size=40, )
        button = Button(text="Start", size=(20, 50), size_hint=(0.1, 0.1), pos_hint={'top': .25, 'right': .50},
                        font_size=20, on_press=self.go_to_login)
        button2 = Button(text="Log out", size=(20, 50), size_hint=(0.1, 0.1), pos_hint={'top': .25, 'right': .64},
                        font_size=20, on_press=self.endup)
        image = Image(source='datap/e.png',size=(1100,600))
        layout.add_widget(image)
        layout.add_widget(button)
        layout.add_widget(button2)
        layout.add_widget(tlabel)
        layout.add_widget(alabel)
       
        self.add_widget(layout)

    def _update_rect(self, instance, value):
        # Update the size and position of the rectangle
        self.rect.size = instance.size
        self.rect.pos = instance.pos

    def go_to_login(self, instance):
        viv()
        pol()
    def endup(self,instance):
        self.manager.current = 'login'

class Admin(Screen):
    def __init__(self, **kwargs):
        super(Admin, self).__init__(**kwargs)
        layout = FloatLayout()
        with layout.canvas.before:
            Color(127, 0, 255, 0.4)  # RGBA values (red, green, blue, alpha)
            self.rect = Rectangle(size=layout.size, pos=layout.pos)
        layout.bind(size=self._update_rect, pos=self._update_rect)
        button = Button(text="Update",size_hint=(0.15, 0.08),pos_hint={'center_x': 0.3, 'center_y': 0.1})
        buttonl = Button(text="Logout",size_hint=(0.15, 0.08),pos_hint={'center_x': 0.5, 'center_y': 0.1})
        buttoncu = Button(text="Change user",size_hint=(0.15, 0.08),pos_hint={'center_x': 0.7, 'center_y': 0.1})
        tlabel = Label(text='Admin Page', pos_hint={'center_x': 0.5, 'center_y': 0.92}, font_size=40)
        slabel=Label(text='Level Above 40 is Stressed', pos_hint={'center_x': 0.5, 'center_y': 0.2}, font_size=20)
        layout.add_widget(tlabel)
        layout.add_widget(slabel)
        layout.add_widget(buttonl)
        layout.add_widget(buttoncu)
        button.bind(on_press=self.update)
        buttonl.bind(on_press=self.go_to_login)
        buttoncu.bind(on_press=self.change)
        layout.add_widget(button)
        self.img = Image(source='plot.png', size_hint=(None, None), size=(600,600), pos_hint={'center_x': 0.71, 'center_y': 0.53})
        self.add_widget(layout)
        self.add_widget(self.img)
        self.file_path = "Final report.txt"  # Path to your text file 
        self.file_contents = TextInput(text=self.read_file(), readonly=True,size_hint=(.4,0.6),pos_hint={'center_x': 0.22, 'center_y': 0.53})
        self.add_widget(self.file_contents)        
        
    def read_file(self):
        if os.path.exists(self.file_path):
            with open(self.file_path, "r") as file:
                return file.read()
        else:
            return "File not found!"

    def _update_rect(self, instance, value):
        # Update the size and position of the rectangle
        self.rect.size = instance.size
        self.rect.pos = instance.pos

    def update(self, instance):
        self.file_contents.text = self.read_file()
        self.file_contents.canvas.ask_update()
        self.img.source = 'plot.png'
        self.img.reload()  # Force a reload of the image to ensure it updates
        self.img.canvas.ask_update()
        print('===========>executed')


    def go_to_login(self, instance):
        self.manager.current = 'login'



    def change(self,instance):
        self.manager.current = 'cuser'

class Changeuser(Screen):
    def __init__(self, **kwargs):
        super(Changeuser, self).__init__(**kwargs)
        alayout = FloatLayout()
        with alayout.canvas.before:
            Color(127, 0, 255, 0.4)  # RGBA values (red, green, blue, alpha)
            self.rect = Rectangle(size=alayout.size, pos=alayout.pos)
        alayout.bind(size=self._update_rect, pos=self._update_rect)
        layout = BoxLayout(orientation='vertical', size_hint=(0.4, 0.4), pos_hint={'x': .3, 'y': .4})
        username_label = Label(text='Username:')
        self.username_input = TextInput(multiline=False)
        password_label = Label(text='Password:')
        self.password_input = TextInput(multiline=False)
        login_buttonu = Button(text='Change user', size_hint=(.30, .15), pos_hint={'x': .15, 'y': .1})
        login_buttonu.bind(on_press=self.changeu)
        login_buttona = Button(text='Change Admin', size_hint=(.30, .15), pos_hint={'x': .6, 'y': .1})
        login_buttona.bind(on_press=self.changea)
        layout.add_widget(username_label)
        layout.add_widget(self.username_input)
        layout.add_widget(password_label)
        layout.add_widget(self.password_input)
        alayout.add_widget(login_buttonu)
        alayout.add_widget(login_buttona)
        alayout.add_widget(layout)
        self.add_widget(alayout)


    def _update_rect(self, instance, value):
        # Update the size and position of the rectangle
        self.rect.size = instance.size
    
    def changea(self,instance):
        text = self.username_input.text
        pw = self.password_input.text
        f=open('datap/admin.txt','w')
        f.write(text+'\n'+pw)
        print(text+'\n'+pw)
        f.close()
        exit()
    def changeu(self,instance):
        text = self.username_input.text
        pw = self.password_input.text
        f=open('datap/user.txt','w')
        f.write(text+'\n'+pw)
        f.close()
        exit()

class MyApp(App):
    def build(self):
        self.title = 'Employee Stress Analysis' 
        Window.size = (1100, 600)
        sm = ScreenManager()
        sm.add_widget(Introduction(name='intro'))
        sm.add_widget(LoginPage(name='login'))
        sm.add_widget(Employee(name='a'))
        sm.add_widget(Admin(name='ad'))
        sm.add_widget(Changeuser(name='cuser'))
        return sm



fu=open('datap/user.txt','r')
u1=fu.readline()
u1=u1.strip()
u2=fu.readline()
u2=u2.strip()
fa=open('datap/admin.txt','r')
a1=fa.readline()
a1=a1.strip()
a2=fa.readline()
a2=a2.strip()
MyApp().run()


