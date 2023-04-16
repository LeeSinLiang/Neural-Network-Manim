# Manim Cairo
from manimlib import *
# ManimGL
# from manimlib import *
import sys
import os.path
from manimlib.mobject.svg.drawings import SpeechBubble
from manimlib.mobject import geometry
# import six,colour,nncc2gl
from nn import *

class Application(NeuralNetwork):
    def construct(self):
        NeuralNetwork.construct(self)
        self.play(
            *[FadeOut(mob)for mob in self.mobjects]
            # All mobjects in the screen are saved in self.mobjects
        )
        NeuralNetwork.arguments['layer_sizes'] = [4,6,6,2]
        NeuralNetwork.add_neurons(self,False)
        NeuralNetwork.add_edges(self,False)
        NeuralNetwork.group1(self)
        self.grp.scale(0.5)
        print('...')
        self.clear()
        lineoguidance = Line([-3, 0, 0], [3, 0, 0]).rotate(90*DEGREES)
        self.wait(6)
        self.v = self.virtual_assistant()
        self.wait(12.5)
        grp=self.grp
        NeuralNetwork.arguments['layer_sizes'] = [4, 6, 6, 2]
        NeuralNetwork.add_neurons(self, False)
        NeuralNetwork.add_edges(self, False)
        NeuralNetwork.group1(self)
        self.grp.scale(0.5)
        v_copy = self.v.copy().move_to(lineoguidance.get_end()).scale(0.18)
        lineoguidance.rotate(60*DEGREES)
        self.email = self.mail().move_to(lineoguidance.get_end()).scale(0.25)
        lineoguidance.rotate(60*DEGREES)
        self.youtube = self.yt_logo().move_to(lineoguidance.get_end()).scale(0.25)
        lineoguidance.rotate(60*DEGREES)
        self.tc = self.textclassification().move_to(lineoguidance.get_end()).scale(0.2)
        lineoguidance.rotate(60*DEGREES)
        self.stock_prediction = self.stock_prediction().move_to(
            lineoguidance.get_end()).scale(0.15)
        lineoguidance.rotate(60*DEGREES)
        self.credit_card = self.credit_card().move_to(lineoguidance.get_end()).scale(0.25)
        # self.rotating_applications()
        # sun = Circle()


        self.play(Transform(self.v, v_copy), FadeIn(self.credit_card), FadeIn(
            self.email), FadeIn(self.tc), FadeIn(self.stock_prediction), FadeIn(self.youtube),ReplacementTransform(grp,self.grp))
        self.path = ArcBetweenPoints(start=self.email.get_center(),end=self.youtube.get_center(),radius=3,angle=TAU/6)
        self.path1 = ArcBetweenPoints(start=self.youtube.get_center(),end=self.tc.get_center(),radius=3,angle=TAU/6)
        self.path2 = ArcBetweenPoints(start=self.tc.get_center(),end=self.stock_prediction.get_center(),radius=3,angle=TAU/6)
        self.path3 = ArcBetweenPoints(start=self.stock_prediction.get_center(),end=self.credit_card.get_center(),radius=3,angle=TAU/6)
        self.path4 = ArcBetweenPoints(start=self.credit_card.get_center(),end=self.v.get_center(),radius=3,angle=TAU/6)
        self.path5 = ArcBetweenPoints(start=self.v.get_center(),end=self.email.get_center(),radius=3,angle=TAU/6)
        # path5 = ArcBetweenPoints(start=credit_card.get_center(),end=tc.get_center(),radius=3,angle=TAU/6)

        self.rotating_applications()
        self.wait(12)

        # self.embed()

    def alt_text(self):
        self.ttc = Text('Text Classification').scale(0.5)
        self.tem = Text('Spam Filters').scale(0.5)
        self.tsp = Text('Stock Prediction').scale(0.5)
        self.cc = Text('Fraud Detection').scale(0.5)
        self.yt = Text('Video Recommendation').scale(0.5)
        self.tv = Text('Virtual Assistant').scale(0.5)

        always(self.ttc.next_to, self.cc_copy, RIGHT)
        always(self.tem.next_to, self.email_copy, RIGHT)
        always(self.tsp.next_to , self.stock_prediction_copy, RIGHT)
        always(self.yt.next_to, self.youtube_copy, RIGHT)
        always(self.cc.next_to, self.cc_copy, RIGHT)
        # always(self.tv.next_to, self.v, RIGHT)

    def rotating_applications(self):

        self.cc_copy = self.credit_card.copy().move_to(3*UP).scale(2)
        self.tc_copy = self.tc.copy().move_to(3*UP).scale(2)
        self.email_copy = self.email.copy().move_to(3*UP).scale(2)
        self.stock_prediction_copy = self.stock_prediction.copy().move_to(3*UP).scale(2)
        self.youtube_copy = self.youtube.copy().move_to(3*UP).scale(2)
        # cc_copy = self.credit_card.copy().move_to(3*UP).scale(2)
        self.alt_text()

        self.play(MoveAlongPath(self.email, self.path),
                    MoveAlongPath(self.youtube, self.path1),
                    MoveAlongPath(self.tc, self.path2),
                    MoveAlongPath(self.stock_prediction, self.path3),
                    MoveAlongPath(self.credit_card, self.path4),
                    MoveAlongPath(self.v, self.path5),
                    Transform(self.credit_card,self.cc_copy))
        self.play(FadeIn(self.cc))
        self.play(ApplyMethod(self.credit_card.scale,0.5),
                    MoveAlongPath(self.email, self.path1),
                    MoveAlongPath(self.youtube, self.path2),
                    MoveAlongPath(self.tc, self.path3),
                    MoveAlongPath(self.stock_prediction, self.path4),
                    MoveAlongPath(self.credit_card, self.path5),
                    MoveAlongPath(self.v,self.path),
                    Transform(self.stock_prediction,self.stock_prediction_copy),
                    ReplacementTransform(self.cc,self.tsp))
        self.wait()
        self.play(ApplyMethod(self.stock_prediction.scale,0.5),
                    MoveAlongPath(self.email, self.path2),
                    MoveAlongPath(self.youtube, self.path3),
                    MoveAlongPath(self.tc, self.path4),
                    MoveAlongPath(self.stock_prediction, self.path5),
                    MoveAlongPath(self.credit_card, self.path),
                    MoveAlongPath(self.v, self.path1),
                    Transform(self.tc,self.tc_copy),
                    ReplacementTransform(self.tsp,self.ttc))
        self.wait()
        self.play(ApplyMethod(self.tc.scale,0.5),
                    MoveAlongPath(self.email, self.path3),
                    MoveAlongPath(self.youtube, self.path4),
                    MoveAlongPath(self.tc, self.path5),
                    MoveAlongPath(self.stock_prediction, self.path),
                    MoveAlongPath(self.credit_card, self.path1),
                    MoveAlongPath(self.v, self.path2),
                    Transform(self.youtube,self.youtube_copy),
                    ReplacementTransform(self.ttc,self.yt))
        self.wait()
        self.play(ApplyMethod(self.youtube.scale,0.5),
                    MoveAlongPath(self.email, self.path4),
                    MoveAlongPath(self.youtube, self.path5),
                    MoveAlongPath(self.tc, self.path),
                    MoveAlongPath(self.stock_prediction, self.path1),
                    MoveAlongPath(self.credit_card, self.path2),
                    MoveAlongPath(self.v, self.path3),
                    Transform(self.email, self.email_copy),
                    ReplacementTransform(self.yt,self.tem))
        self.wait()


    def full_rotate_without_zoom(self,va,v):
        lineoguidance = Line([-3, 0, 0], [3, 0, 0]).rotate(60*DEGREES)
        credit_card = self.credit_card().move_to(lineoguidance.get_end()).scale(0.25)
        lineoguidance.rotate(60*DEGREES)
        tc = self.textclassification().move_to(lineoguidance.get_end()).scale(0.2)
        lineoguidance.rotate(120*DEGREES)
        email = self.mail().move_to(lineoguidance.get_end()).scale(0.25)
        stock_prediction = self.stock_prediction().move_to(ORIGIN+3*LEFT).scale(0.1)
        lineoguidance.rotate(60*DEGREES)
        youtube = self.yt_logo().move_to(lineoguidance.get_end()).scale(0.25)
        lineoguidance.rotate(60*DEGREES)
        c = Circle()
        c1 = Circle()
        c2 = Circle()
        c3 = Circle()
        c4 = Circle()
        c.move_to(credit_card.get_center())
        c1.move_to(email.get_center())
        c2.move_to(tc.get_center())
        c3.move_to(stock_prediction.get_center())
        c4.move_to(youtube.get_center())
        group1 = VGroup(credit_card, c)
        group2 = VGroup(email, c1)
        group3 = VGroup(tc, c2)
        group4 = VGroup(stock_prediction, c3)
        group5 = VGroup(youtube, c4)
        path1 = Circle(radius=3).rotate(60*DEGREES)
        path2 = Circle(radius=3).rotate(120*DEGREES)
        path3 = Circle(radius=3).rotate(180*DEGREES)
        path4 = Circle(radius=3).rotate(240*DEGREES)
        path5 = Circle(radius=3).rotate(300*DEGREES)
        self.add(group1, group2, group3, group4, group5)

        #path = ArcBetweenPoints(start=tc.get_center(),end=stock_prediction.get_center(),radius=3,angle=TAU/6)
        # infogroup1 = VGroup(v,infocircle1)
        self.play(MoveAlongPath(v.infogroup1, v.path), MoveAlongPath(group1, path1), MoveAlongPath(group2, path2), MoveAlongPath(
            group3, path3), MoveAlongPath(group4, path4), MoveAlongPath(group5, path5), run_time=4, rate_func=linear)
        # self.remove(v)
        Arc()
        # v = self.textclassification()
        # self.add(v)
        # credit_card = self.credit_card()
        # self.play(ShowCreation(credit_card))
        return None



    def virtual_assistant(self):
        r = RoundedRectangle(height=5, width=3).shift(4*RIGHT)
        self.grp.scale(1.5).next_to(r,LEFT).shift(0.5*LEFT)
        person = SVGMobject('assets/user(3).svg').to_edge(LEFT)
        person.set_stroke(width=0.2)
        bubble = SpeechBubble(height=1.5, width=2.5)
        bubble.add_content(Text('Hey Google', color=WHITE))
        # bubble.set_fill(opacity=0)
        bubble.set_stroke(WHITE, 1)
        # bubble.pin_to(self)
        message = Rectangle(height=1, width=2.5).shift(RIGHT)
        message.move_to(r.get_center()+DOWN)
        t = Text('Hi, How can I help?').scale(
            0.25).move_to(message.get_center())
        g = Text('G').next_to(r.get_corner(RIGHT+UP), LEFT+DOWN)
        self.play(ShowCreation(r))
        self.play(Write(person))
        VGroup(bubble, bubble.content).next_to(
            person.get_corner(RIGHT+UP), RIGHT)
        self.play(ShowCreation(bubble), Write(bubble.content))
        self.wait(2)
        self.play(ShowCreation(message), Write(t), Write(g))
        self.wait(10)
        self.play(Write(self.grp))
        self.wait()
        v = VGroup(r, person, bubble, bubble.content, message, t, g)
        return v

    def mail(self):
        return SVGMobject('assets/mail.svg')

    def applications_example(self):
        example1 = Text('Text Recongition')
        example2 = Text('Image Recongition')
        example3 = Text('Speech Recongition')
        example4 = Text('Text Recongition')

    def credit_card(self):
        card = RoundedRectangle(height=3, width=5)
        upper_line = Rectangle(height=0.2, width=5).set_fill(WHITE, 1)
        upper_line.shift(0.7*UP)
        line1 = Line(stroke_width=10).set_fill(WHITE, 1)
        line2 = Line(stroke_width=10).set_fill(WHITE, 1)
        line3 = Line(stroke_width=10).set_fill(WHITE, 1)
        line4 = Line(stroke_width=10).set_fill(WHITE, 1)
        line1.set_length(0.8)
        line2.set_length(0.8)
        line3.set_length(0.8)
        line4.set_length(0.8)
        line_bottom = Line(stroke_width=10)
        line1.shift(1.8*LEFT)
        line2.shift(1.8*RIGHT)
        line3.shift(0.6*RIGHT)
        line4.shift(0.6*LEFT)
        line_bottom.move_to(card.get_center()+0.15*BOTTOM+1.2*LEFT)
        credit_card = VGroup(card, upper_line, line1,
                             line2, line3, line4, line_bottom)
        return credit_card

    def yt_logo(self):
        t = Triangle().set_fill(WHITE, 1)
        t.rotate(270*DEGREES, about_point=np.array([1, 1, 0]))
        t.move_to(ORIGIN)
        r = RoundedRectangle(height=3, width=5)
        r.set_fill(RED, 1)
        r.set_stroke(RED,1)
        t.set_fill(WHITE,1)
        real_yt_logo = VGroup(r,t)
        return real_yt_logo

        # self.embed()

    def textclassification(self):
        box = Rectangle(height=5).shift(LEFT)
        t = Text('Article').scale(0.6).shift(LEFT)
        box.surround(t)
        pointer1 = CurvedArrow(ORIGIN,UP+2*RIGHT,angle=-TAU/4)
        pointer2 = Arrow(ORIGIN,2*RIGHT)
        pointer3 = CurvedArrow(ORIGIN,DOWN+2*RIGHT)
        t1 = Text('Science').scale(0.6).shift(UP+3.5*RIGHT)
        t2 = Text('Buisness').scale(0.6).shift(3.5*RIGHT)
        t3 = Text('Entertainment').scale(0.6).shift(DOWN+4*RIGHT)
        tc = VGroup(box,t,pointer1,pointer2,pointer3,t1,t2,t3)
        return tc

    def stock_prediction(self):
        #random values here
        x = [1, 3, 5, 7, 9, 11]
        y = [3, 6, 1, 7, 4, 9]
        axes = Axes((0,12),(0,12))
        dot_collection = VGroup()
        for time, dat in zip(x,y):
            dot = Dot().move_to(axes.coords_to_point(time , dat))
            dot_collection.add(dot)
        line_collection = VGroup()
        for i in range(5):
            line = Line(dot_collection[i].get_center(),dot_collection[i+1].get_center())
            line_collection.add(line)

        return VGroup(axes,dot_collection,line_collection)

    def title(self):
        return

'''
1. Text classificaon /
2. Email spam filters /
3. Speech Recognition /
4. Recommender System /
5. Fraud Detection /
6. Virtual Assistant /
'''
