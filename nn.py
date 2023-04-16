from manimlib import *
import random
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
red1 = "#FF0000"
blue1 = "#0000ff"
light_grey1 = "#6b6b78"
green1 = "#00FF00"
violet1 = "#FF00FF"


class NeuralNetwork(Scene):
    arguments = {
        "network_size": 1,
        "network_position": ORIGIN,
        "layer_sizes": [6, 12, 10, 8, 6, 2],
        "layer_buff": 1.25*LARGE_BUFF,
        # "layer_buff": LARGE_BUFF,
        "neuron_radius": 0.15,
        "neuron_color": WHITE,
        "neuron_width": 2,
        "neuron_fill_color": BLACK,
        "neuron_fill_opacity": 1,
        "neuron_buff": MED_SMALL_BUFF,
        "edge_color": YELLOW_B,
        "edge_width": 2,
        "edge_opacity": 1,
        "layer_label_color": WHITE,
        "layer_label_size": 0.5,
        "neuron_label_color": WHITE,
    }

    # def __init__(self):
    #     self.nPclass = 0
    #     self.nSex = 0
    #     self.nAge = 0
    #     self.nSibSp = 0
    #     self.Fare = 0
    #     self.Embarked = 0
    #     self.Age_was_missing = 0
    # def setup(self):
    #     MovingCameraScene.setup(self)

    def construct(self):
        # self.camera_frame.save_state()

        self.add_neurons()
        self.add_edges()
        # self.label_stuff()
        self.group1()
        # self.preload()
        # self.grp.n = self.n
        # self.grp.output_label = self.output_label
        # self.grp.add(self.n)
        # self.grp.add(self.output_label)
        self.intro()
        self.grp.n = self.n
        self.grp.add(self.n)

        # self.content()
        # grp = VGroup()
        # grp.layers = self.layers
        # grp.edges = self.edge_groups
        # grp.n = self.n
        # grp.output_label = self.output_label
        # grp.add(self.layers)
        # grp.add(self.edge_groups)
        # grp.add(self.n)
        # grp.add(self.output_label)
        # grp.shift(LEFT)
        # self.play(ApplyMethod(grp.scale,0.72))
        # self.label_layers()
        # box = Rectangle(height=30, color=YELLOW, stroke_width=4)
        # box.surround(self.n)
        # self.play(ShowCreation(box))
        # grp1 = VGroup()
        # grp1.add(self.layers[1][0])
        # grp1.add(self.layers[2][0])
        # grp1.add(self.layers[3][0])
        # grp1.add(self.layers[4][0])
        # grp1.add(self.output_label)
        # bland = Rectangle(height=20,fill_color=BLACK, fill_opacity=0.4, stroke_opacity=0)
        # bland.surround(grp1)
        # self.play(ShowCreation(bland))
        return self.grp

    def group1(self):
        grp = VGroup()
        grp.layers = self.layers
        grp.edges = self.edge_groups
        grp.add(self.layers)
        grp.add(self.edge_groups)
        self.grp = grp
        return grp

    def intro(self):
        grp1 = self.grp.copy().scale(0.8).shift(3*RIGHT)
        # self.play(FadeIn(self.grp))
        # self.play(ShowCreation(grp1))
        self.play(Transform(self.grp, grp1))
        neural = TexText("Neural")
        title = VGroup(
            neural,
            TexText("Network Crash"),
            TexText("Course")
        ).arrange(DOWN, aligned_edge=LEFT).shift(3.5*LEFT).scale(1.1)
        line1 = Line(title.get_corner(UP+LEFT), neural.get_corner(UP+RIGHT),
                     color=YELLOW_E, stroke_width=5).next_to(title, UP).shift(LEFT)
        line = Line(title.get_corner(DOWN+LEFT), title.get_corner(DOWN+RIGHT),
                    color=YELLOW_E, stroke_width=5).next_to(title, DOWN)
        self.play(Write(title), ShowCreation(line), ShowCreation(line1))
        ml = Text('Machine Learning:', font="consolas", font_size=24)
        self.quote = VGroup(
            ml,
            Text("Programming with examples,", font="consolas",
                 font_size=24, t2c={"examples": YELLOW}),
            Text("not instructions.", font="consolas",
                 font_size=24, t2c={"instructions": YELLOW}),
            Tex('-\\ Kyle\\ Mcdonald')
        ).arrange(DOWN, aligned_edge=LEFT).to_edge(LEFT).shift(0.5*RIGHT)

        # mob1.align_to(mob2, alignment_vect = RIGHT)
        self.introgrp = VGroup(title, line, line1)
        title.add(line1)
        title.add(line)
        self.title = title
        self.animation(self.grp)

    def add_neurons(self, animation=True):
        self.loop = False
        layers = VGroup(*[self.get_layer(size)
                        for size in NeuralNetwork.arguments["layer_sizes"]])
        layers.arrange(RIGHT, buff=NeuralNetwork.arguments["layer_buff"])
        layers.scale(NeuralNetwork.arguments["network_size"])
        # self.layers is layers, but we can use it throughout every method in our class
        # without having to redefine layers each time
        self.layers = layers
        layers.shift(NeuralNetwork.arguments["network_position"])
        self.play(FadeInFromPoint(layers, ORIGIN),
                  run_time=2) if animation is True else None
        # self.add(layers)

    def get_layer(self, size):
        layer = VGroup()
        n_neurons = size
        neurons = VGroup(*[
            Circle(
                radius=NeuralNetwork.arguments["neuron_radius"],
                stroke_color=NeuralNetwork.arguments["neuron_color"],
                stroke_width=NeuralNetwork.arguments["neuron_width"],
                fill_color=NeuralNetwork.arguments["neuron_fill_color"],
                fill_opacity=NeuralNetwork.arguments["neuron_fill_opacity"],
            )
            for i in range(n_neurons)
        ])
        neurons.arrange(DOWN, buff=NeuralNetwork.arguments["neuron_buff"])
        layer.neurons = neurons
        layer.add(neurons)
        if size == 6 and self.loop == False:
            self.neurons1 = neurons
            self.loop = True
        return layer

    def edge_security(self):
        self.edge_groups = VGroup()
        for l1, l2 in zip(self.layers[:-1], self.layers[1:]):
            edge_group = VGroup()
            for n1, n2 in it.product(l1.neurons, l2.neurons):
                edge = self.get_edge(n1, n2)
                edge_group.add(edge)
            self.edge_groups.add(edge_group)

    def add_edges(self, animation=True, color=True):
        self.edge_groups = VGroup()
        for l1, l2 in zip(self.layers[:-1], self.layers[1:]):
            edge_group = VGroup()
            for n1, n2 in it.product(l1.neurons, l2.neurons):
                # print(x,y)
                edge = self.get_edge(n1, n2, color)
                edge_group.add(edge)

            # edge_group = self.set_bias(x,edge_group) if x is not len(NeuralNetwork.arguments['layer_sizes'])-2 else None
            self.play(Write(edge_group),
                      run_time=0.5) if animation is True else None
            self.edge_groups.add(edge_group)

    def get_edge(self, neuron1, neuron2, color):
        colors = [red1, blue1, blue1]
        r = random.randint(0, len(colors)-1)
        # if y % self.arguments['layer_sizes'][x+1] == 0 and x is not len(self.arguments['layer_sizes'])-2:
        #     print(x,y)
        #     color = GREY
        # else:
        #     color = colors[r]
        # print(y%NeuralNetwork.arguments['layer_sizes'][x+1] != 0,x,y)
        return Line(
            neuron1.get_center(),
            neuron2.get_center(),
            # buff=1.25*NeuralNetwork.arguments["neuron_radius"],
            buff=NeuralNetwork.arguments["neuron_radius"],

            stroke_color=GREY if color is False else colors[r],
            stroke_width=NeuralNetwork.arguments["edge_width"],
            stroke_opacity=NeuralNetwork.arguments["edge_opacity"]
        )

    def set_bias(self, x):
        x1 = 0
        # return None
        for i in range(self.arguments['layer_sizes'][x]):
            print(x, x1)
            self.edge_groups[x][x1].set_color(GREY)
            x1 += self.arguments['layer_sizes'][x+1]
        # self.edge_groups[0][0].set_color(GREY)

    def animation(self, grop, backward_teaser=False, L=1):
        input_data1 = self.read_data('test.csv', False)
        # print(input_data)
        output = self.read_data('my_submission.csv', False)

        # output_data = self.read_data('ans.csv',False)
        # for i in range(len(input_data1)):
        # y=None
        if backward_teaser is True:
            y = np.array(pd.read_csv(
                'data/without_training/output_real.csv').T.reset_index().drop('index', axis=1)[0])
            y_hat = np.array(pd.read_csv(
                'data/without_training/output.csv').T.reset_index().drop('index', axis=1)[0])
            p = np.zeros((1, y_hat.shape[0]))
            for i in range(0, y_hat.shape[0]):
                if y_hat[i] > 0.5:
                    p[0, i] = 1
                else:
                    p[0, i] = 0
            print(y[0])
            self.ans = Tex(str(y[1])).next_to(
                self.grp.layers[len(self.arguments['layer_sizes'])-1], RIGHT).shift(RIGHT)

        self.activate = False
        ans_output = False
        # always(self.ans.next_to,
        #     self.grp.layers[len(self.arguments['layer_sizes'])-1], 2*RIGHT)
        for i in range(L):
            self.input_data(input_data1.T[i], True, backward_teaser=True, y=y, i=i+1) if backward_teaser is True else self.input_data(
                input_data1.T[i], True)

            # self.wait()
            for i1 in range(4):
                df = self.read_data('activation'+str(i1+1)+'.csv',
                                    True) if backward_teaser is False else self.read_data('without_training/activation'+str(i1+1)+'.csv', True)
                # for x in range(len(df.T)):
                #     print(df[x][i])
                self.play(*[ApplyMethod(grop.layers[i1+1][0][x].set_fill,
                                        WHITE, float(df[x][i])) for x in range(len(df.T))])
                if i1 == 1 and backward_teaser is False:
                    print("DEY")
                    self.play(TransformMatchingParts(
                        self.title, self.quote, transform_mismatches=True))

            print(output['Survived'][i], 6)
            self.output_data(output['Survived'][i], len(self.arguments['layer_sizes'])-1) if backward_teaser is False else self.output_data(
                int(list(p[0])[i+1]), len(self.arguments['layer_sizes'])-1, y[i+1])
            if backward_teaser is True:
                # self.wait()
                self.grp.layers.set_fill(WHITE, 0)
                self.play(FadeOut(self.fade_out))
            if backward_teaser is True and i == 1:
                accuracy = TexText('Accuracy: 38\%').move_to([4, 3, 0])
                self.play(Write(accuracy)) if ans_output is False else None
                ans_output = True
                self.wait()
                weights_random = TexText('Weights are initialized randomly!').next_to(
                    self.grp.layers[2], UP).to_edge(LEFT).shift(RIGHT+0.5*DOWN)
                grp = self.grp.copy()
                grp.shift(DOWN).scale(0.8)
                n = self.n.copy()
                for i in range(6):
                    n[i].move_to(self.grp.layers[0][0][i].get_center())
                self.play(self.grp.animate.shift(DOWN).scale(0.8), accuracy.animate.shift(2.5*DOWN), self.ans.animate.next_to(
                    grp.layers[len(self.arguments['layer_sizes'])-1], RIGHT).shift(RIGHT), Transform(self.n, n))
                self.play(Write(weights_random))
                self.play(
                    Write(TexText('= Guessing Game!').next_to(weights_random, RIGHT)))
        #self.play( self.camera_frame.scale,0.05,self.camera_frame.move_to,self.layers[0][0][0])
        #self.wait()

    def preload(self):
        input_data1 = self.read_data('test.csv', False)
        self.input_data(input_data1.T[0], False)
        print(1)

    def label_layers(self):

        braces = Brace(self.layers[0], LEFT)
        self.t = braces.get_text("Input Layer").scale(0.8).shift(0.5*RIGHT)
        # t = TexText("Input Layer", tex_to_color_map={'Input': WHITE}).next_to(braces,LEFT).scale(0.8)
        self.play(GrowFromCenter(braces), Write(self.t))
        self.wait()
        grp = VGroup()
        # for i in range(4):
        #     grp.add(self.layers[i+1][0][0].get_center())
        line = Line(self.grp.layers[1][0][0].get_center(
        ), self.grp.layers[4][0][0].get_center())

        braces1 = Brace(line, direction=line.copy().rotate(
            PI / 2).get_unit_vector())
        t1 = braces1.get_text("Hidden Layer").scale(0.8).shift(0.5*RIGHT)

        self.play(GrowFromCenter(braces1), Write(t1))
        braces2 = Brace(self.grp.output_label, RIGHT)
        t2 = braces2.get_text("Output Layer").scale(0.8).shift(0.5*LEFT)
        self.play(GrowFromCenter(braces2), Write(t2))

    def read_data(self, fil, activation):
        foo = pd.read_csv('data/'+fil)
        if activation == True:
            scaler = MinMaxScaler()
            d = scaler.fit_transform(foo.T)
            scaled_df = pd.DataFrame(d)
            scaled_df, scaled_df_selected = train_test_split(
                scaled_df, test_size=0.1, random_state=42)

        else:

            scaled_df, scaled_df_selected = train_test_split(
                foo, test_size=0.1, random_state=42)

        return scaled_df_selected.reset_index().drop('index', axis=1)

    def input_data(self, a, animate, backward_teaser=False, y=None, i=None):

        n = VGroup()
        for s in a:
            t = Tex(str(round(int(s), 2)))
            n.add(t)
        n.arrange(DOWN, buff=0.3)
        n.next_to(self.grp.layers, LEFT)
        ans = Tex(str(y[i])).next_to(
            self.grp.layers[len(self.arguments['layer_sizes'])-1], RIGHT).shift(RIGHT) if backward_teaser is True else None
        print(ans, 'ans') if backward_teaser is True else None
        n1 = n.copy()
        n1.scale(0.2)
        for i in range(6):
            n1[i].move_to(self.neurons1[i].get_center())
        if animate is True:
            try:
                self.play(FadeOut(self.n))
            except:
                pass
            print('hello')
            if backward_teaser is True:
                self.play(Transform(n, n1), Write(self.ans)) if self.activate is False else self.play(
                    Transform(n, n1), Transform(self.ans, ans))
                self.activate = True
            else:
                self.play(Transform(n, n1))
            self.n = n
        else:
            self.n = n1

    def label_stuff(self):
        a = [str(random.randint(1, 10)) for i in range(2)]
        n = VGroup()
        survived = Tex('Survive').scale(0.8)
        not_survived = Tex('Not Survive').scale(0.8)
        n.add(survived)
        n.add(not_survived)
        n.arrange(DOWN, buff=MED_SMALL_BUFF)
        n.next_to(self.layers, RIGHT)
        self.play(Write(n))
        title = TexText("Neural Network of Titanic Dataset",
                        tex_to_color_map={'Neural': WHITE}).to_edge(UP)
        self.play(Write(title))
        self.output_label = n

    def output_data(self, a, layer, y=None):
        print(a, layer, 'output')
        if self.arguments['layer_sizes'][-1] != 1 and y is None:
            self.layers[layer][0][0].set_fill(WHITE, 0)
            self.layers[layer][0][1].set_fill(WHITE, 0)
            self.play(ApplyMethod(self.layers[layer][0][a].set_fill, WHITE, 1))
        elif self.arguments['layer_sizes'][-1] is 1 and y is not None:
            not_equal = Tex('!=', color=RED).next_to(
                self.grp.layers[len(self.arguments['layer_sizes'])-1], RIGHT)
            equal = Tex('==', color=GREEN).next_to(
                self.grp.layers[len(self.arguments['layer_sizes'])-1], RIGHT)
            self.play(ApplyMethod(self.layers[layer][0][0].set_fill, WHITE, a))
            self.play(Write(not_equal)) if a != y else self.play(
                Write(equal))
            self.wait()
            self.fade_out = not_equal if a != y else equal

    def content(self):
        content = VGroup(
            TexText("Course Overview:"),
            TexText("1. What is Neural Network?"),
            TexText("2. Neuron"),
            TexText('3. Weights,'),
            TexText('Bias and Connections').shift(RIGHT),
            TexText("4. Layers")
        ).arrange(DOWN, aligned_edge=LEFT).to_edge(UP+LEFT)
        self.grp.add(self.t)

        self.play(ReplacementTransform(self.t, content))
        self.grp.remove(self.t)
