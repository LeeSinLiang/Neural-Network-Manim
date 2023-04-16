# Manim Cairo
from manimlib import *
# ManimGL
# from manimlib import *
import sys
import matplotlib.pyplot as plt
import os.path
from manimlib.mobject.svg.drawings import SpeechBubble
from manimlib.mobject import geometry
# import six,colour,nncc2gl
from nn import *
from applications import *
from bias_update import *
from backward_teaser import *
from graph1 import *
from endscreen import *
class Explanationall(EndScreen):
    def construct(self):
        # self.title = TexText('Test').to_edge(UP+LEFT)
        NeuralNetwork.construct(self)
        self.fade_all_except_title()
        Application.construct(self)
        self.fade_all_except_title()
        self.wait(2)
        NeuralNetwork.arguments['layer_sizes'] = [4,6,6,2]
        NeuralNetwork.add_neurons(self,False)
        NeuralNetwork.add_edges(self,False)
        NeuralNetwork.group1(self)
        self.grp.scale(1.5)
        self.edges=self.grp.edges.copy().set_color(YELLOW)
        self.first_part()
        self.play(self.camera.frame.animate.scale(10).move_to(ORIGIN),FadeIn(self.grp.edges),FadeOut(VGroup(self.decimal)))
        self.play(Indicate(self.grp.layers[0],color=WHITE),run_time=3)
        self.wait(3)
        # self.play(ApplyMethod(self.camera.frame.move_to,ORIGIN))
        # self.add(self.grp.scale(1.5))

        self.edge_transform_1 = VGroup()
        self.connections(0, 0, 6,wait=4, highlight=True)
        self.play(Indicate(self.grp.layers[2],color=WHITE))
        self.wait(2)
        self.play(FadeOut(self.grp))
        self.weights()
        # self.wait()

        # self.shallow_nn_representation()
        # self.grp.scale(1.5)
        NeuralNetwork.arguments['layer_sizes'] = [4,6,6,2]
        NeuralNetwork.add_neurons(self,False)
        NeuralNetwork.add_edges(self,False)
        NeuralNetwork.group1(self)
        self.grp.scale(1.5)
        bias=TexText('What is bias?').to_corner(UP+LEFT)
        input_label = self.label("x",VGroup())
        self.play(Write(VGroup(self.grp)),Transform(self.title,bias))


        self.bias_layer = VGroup(*[
            Circle(
                radius=NeuralNetwork.arguments["neuron_radius"],
                stroke_color=NeuralNetwork.arguments["neuron_color"],
                stroke_width=NeuralNetwork.arguments["neuron_width"],
                fill_color=NeuralNetwork.arguments["neuron_fill_color"],
                fill_opacity=NeuralNetwork.arguments["neuron_fill_opacity"],
            ).scale(1.5).next_to(self.grp.layers[x][0][0],UP)
            for x in range(len(NeuralNetwork.arguments['layer_sizes'])-1)

            ]
        )
        bias_label = self.label("b",VGroup(),bias=True)
        self.play(GrowFromPoint(self.bias_layer,self.bias_layer.get_center()),Write(bias_label))
        bias_edges = VGroup()
        x=0
        for l1, l2 in zip(self.grp.layers[:-1], self.grp.layers[1:]):
            edge_group = VGroup()

            y=0
            for n2 in (l2.neurons):
                # print(x,y)
                if y < self.arguments['layer_sizes'][x+1] and x is not len(self.arguments['layer_sizes'])-1:edge_group.add(self.get_edge(self.bias_layer[x], n2, False))
                else:break
                y+=1


            # edge_group = self.set_bias(x,edge_group) if x is not len(NeuralNetwork.arguments['layer_sizes'])-2 else None
            self.play(Write(edge_group),
                      run_time=0.5)
            bias_edges.add(edge_group)
            x+=1
        self.wait(3)
        self.play(self.bias_layer.animate.set_fill(WHITE,1))
        self.wait(5)
        input_grp = VGroup()
        for x in range(NeuralNetwork.arguments['layer_sizes'][0]):
            label = Tex('0')
            label.move_to(self.grp.layers[0][0][x])
            input_grp.add(label)
        self.play(Write(input_grp))
        self.wait(10.5)

        grp = VGroup(self.grp,self.bias_layer,bias_edges,bias_label)
        self.play(grp.animate.shift(DOWN),input_grp.animate.shift(DOWN))
        input_label.shift(DOWN)
        v = VGroup(input_label,bias_label[0]).copy()
        forward_propagation = Tex("{{w_0\cdot x_1}}+{{w_1\cdot x_2}}+{{...}}+{{w_3\cdot x_4+b_0}}").shift(2.5*UP)
        z_forward_propagation = Tex(
            '\sigma({{w_0\cdot x_1}}+{{w_1\cdot x_2}}+{{...}}+{{w_3\cdot x_4+b_0}})').shift(2.5*UP)
        self.play(ReplacementTransform(input_grp, input_label))
        self.wait()
        without_z = Tex("\sum_{i=1}^nw_ix_i+b").shift(3*UP)
        z = Tex('z = ').next_to(without_z,LEFT)
        equal = Tex("=").next_to(without_z,RIGHT)
        inf = Tex("\infty").next_to(equal,RIGHT)
        negativeinf = Tex("-\infty").next_to(equal,RIGHT)
        how = TexText('How?').next_to(inf,RIGHT).shift(RIGHT)
        self.play(ReplacementTransform(v,forward_propagation))
        self.wait(2)
        self.play(Transform(forward_propagation,z_forward_propagation))
        self.wait(5)
        self.play(ReplacementTransform(forward_propagation,without_z))
        self.wait(7)
        self.play(Write(z))
        self.wait(2)
        self.play(FadeIn(VGroup(equal,negativeinf)))
        self.wait()
        self.play(ReplacementTransform(negativeinf,inf))
        self.wait(5)
        # #TODO: activation function: z=wx+b, the values can go up to -inf, +inf, it knows no bounds. so how does the neural network determine which neuron is activated? Thats where activation functions comes in.
        self.play(FadeOut(VGroup(inf,self.grp,without_z,z,equal,input_label,bias_edges,bias_label,self.bias_layer)))
        # self.embed()
        self.wait()
        af = TexText('What is activation function?').to_corner(UP+LEFT)
        a = TexText('Activation Function').shift(3*LEFT)
        arrow = Arrow([0,0,0],[1,0,0]).next_to(a,RIGHT)
        ans = TexText('Output of a neuron').next_to(arrow,RIGHT)
        self.play(Write(a), Transform(self.title, af))
        self.play(Write(arrow))
        self.play(Write(ans))
        v = VGroup(a, arrow, ans)
        self.play(v.animate.shift(2.5*UP))
        self.wait()
        circle = Circle(radius=1.5,color=BLUE).shift(DOWN)
        r_v = ValueTracker(0)
        restricted_value = DecimalNumber(0, num_decimal_places=3, include_sign=False, unit=None)
        restricted_value.add_updater(lambda d: d.set_value(r_v.get_value()))
        restricted_value.move_to(circle.get_center())
        limit = TexText('Limit: -1 to 1').move_to([-2.5,1,0])

        self.play(ShowCreation(circle),FadeIn(restricted_value),Write(limit))
        self.play(r_v.set_value, -1,run_time=2.5)
        self.wait()
        self.play(r_v.set_value, 1,run_time=2.5)
        # self.wait(2)
        print('--')
        self.play(FadeOut(VGroup(circle,restricted_value,limit)))
        self.remove(r_v)

        axes = Axes((-5,5),(0,1)).scale(0.5)
        sigmoid_graph = axes.get_graph(
                    lambda x: 1.0 / (1.0 + math.exp(-x)),
                    use_smoothing=False,
                    color=YELLOW,
                    )
        self.play(Write(axes, lag_ratio=0.01, run_time=1))
        non_linear_label = axes.get_graph_label(sigmoid_graph, Text("Non Linear Function").scale(0.4))
        self.play(
            ShowCreation(sigmoid_graph),
            FadeIn(non_linear_label, RIGHT),
        )
        axes1 = Axes(x_range=[-1, 5],
                     y_range=[-1, 5],width=10).scale(0.4).to_edge(UP+RIGHT)
        linear_graph = axes1.get_graph(lambda x: x,
                               x_range=[-1, 4], color=YELLOW)
        linear_label = axes1.get_graph_label(
            linear_graph, Text("Linear Function").scale(0.3))
        self.wait(10)
        self.play(FadeOut(VGroup(sigmoid_graph,non_linear_label,axes,v)))
        grp.shift(0.8*UP+0.5*LEFT)
        cat = ImageMobject('assets/cat.png').next_to(self.grp.layers[0],LEFT).scale(0.5)
        self.play(Write(self.grp),
                    FadeIn(cat),
                    Write(axes1, lag_ratio=0.01, run_time=1),
                    ShowCreation(linear_graph),
                    FadeIn(linear_label, RIGHT),run_time=3
        )
        soundwave = SVGMobject(
            'assets/soundwavee.svg').scale(0.5).next_to(self.grp.layers[0], LEFT)

        cross = VGroup(Line([3,2,0],[-3,-2,0],color=RED),Line([-3,2,0],[3,-2,0],color=RED)).move_to(self.grp.get_center())
        self.play(FadeTransform(cat,self.dot()),ShowCreation(cross),run_time=2)
        self.wait(5)
        self.play(FadeIn(soundwave))
        self.play(FadeTransform(
            soundwave, self.dot()), ShowCreation(cross),run_time=2)
        self.wait(5)
        self.play(
            *[FadeOut(mob)for mob in self.mobjects]
            # All mobjects in the screen are saved in self.mobjects
        )
        self.title = TexText('Types of Activation Function').to_edge(UP+LEFT)
        self.play(Write(self.title))
        graph_scene.construct(self)
        self.wait()
        self.fade_all_except_title()
        axes = Axes((-5, 5), (0, 1.2),
                    y_axis_config={
                        "decimal_number_config": {
                            "num_decimal_places": 1,
                        }
                    },
                    width=10).scale(0.6).shift(0.5*UP)
        sigmoid_graph = axes.get_graph(
                    lambda x: 1.0 / (1.0 + math.exp(-x)),
                    use_smoothing=False,
                    color=YELLOW,
                    )
        axes1 = Axes((-3, 5), (0, 5),
                     y_axis_config={
            "decimal_number_config": {
                "num_decimal_places": 1,
            }
        },
            width=10).scale(0.6).shift(0.5*DOWN)
        relu_graph = axes1.get_graph(
            lambda x: max(0, x),
            use_smoothing=False,
            color=YELLOW,
        )
        axes2 = Axes((-5, 5), (-1.2, 1.2),
                     y_axis_config={
                        "decimal_number_config": {
                            "num_decimal_places": 1,
                        }
                    },
                    width=10).scale(0.6).shift(0.5*DOWN)
        tanh_graph = axes2.get_graph(
            lambda x: (math.exp(x) - math.exp(-x)) / (math.exp(x) + math.exp(-x)),
            use_smoothing=False,
            color=YELLOW,
        )

        self.play(Write(axes, lag_ratio=0.01, run_time=1))
        non_linear_label = axes.get_graph_label(sigmoid_graph, Text("Sigmoid(Logistic) Function").scale(0.4))
        sigmoid_formula = axes.get_graph_label(
            sigmoid_graph, Tex("\\frac{1}{1+e^{-z}}").scale(1.3),direction=RIGHT).shift(2*DOWN+0.5*RIGHT)
        # sigmoid_formula = Tex("\\frac{1}{1+e^{-x}}").next_to(axes,DOWN).scale(1.3)
        lbl = axes.y_axis.get_number_mobject(0.5,font_size=24)
        tip = Line([-0.1,0.5,0],[0.1,0.5,0])
        axes.add_coordinate_labels(x_values=[-5,5],y_values=[])
        self.play(
            ShowCreation(sigmoid_graph),
            FadeIn(non_linear_label, RIGHT),
            Write(sigmoid_formula),
            Write(lbl),
            Write(tip),
        )
        self.wait(6)
        zero = axes.x_axis.get_number_mobject(0,font_size=24)
        one = axes.y_axis.get_number_mobject(1,font_size=24)
        self.play(Write(zero))
        self.play(Write(one))
        self.wait(4)
        v_grp = VGroup(sigmoid_graph,non_linear_label,sigmoid_formula,lbl, tip,axes)
        v_grp1 = v_grp.copy().shift(4*LEFT).scale(0.8)
        self.wait()
        NeuralNetwork.arguments['layer_sizes'] = [4,6,6,2]
        NeuralNetwork.add_neurons(self,False)
        NeuralNetwork.add_edges(self,False)
        NeuralNetwork.group1(self)
        self.grp.shift(3*RIGHT)
        self.play(Transform(v_grp,v_grp1),FadeOut(VGroup(zero,one)))
        self.play(Write(self.grp))
        explain_binary_classification = TexText('(Classify a set of elements into 2 groups)').next_to(self.grp,DOWN).scale(0.8)
        binary_classification = TexText('Binary Classification').next_to(self.grp,UP).scale(0.8)
        cat_label = TexText('Cat').next_to(self.grp.layers[3][0][0]).scale(0.8)
        dog_label = TexText('Dog').next_to(self.grp.layers[3][0][1]).scale(0.8)
        cat = ImageMobject('assets/cat.png').next_to(self.grp.layers[0],LEFT).scale(0.4).shift(RIGHT*0.8)
        self.play(Write(binary_classification))
        self.play(Indicate(self.grp.layers[3]), Write(
            VGroup(cat_label, dog_label)), FadeIn(cat))
        self.wait()
        self.play(Write(explain_binary_classification))
        self.wait()
        x = [[random.uniform(0, 1) for x1 in range(6)], [
                random.uniform(0, 1) for x1 in range(6)]]
        self.play(FadeTransform(cat,self.dot()))
        self.play(*[ApplyMethod(self.grp.layers[1][0][x1].set_fill,
                                WHITE, float(x[0][x1])) for x1 in range(6)])
        self.wait()
        self.play(*[ApplyMethod(self.grp.layers[2][0][x2].set_fill,
                                WHITE, float(x[1][x2])) for x2 in range(6)])
        self.wait()
        self.play(ApplyMethod(self.grp.layers[3][0][0].set_fill, WHITE, 1))
        self.wait()
        relu_label = axes1.get_graph_label(relu_graph, Text("ReLU Function").scale(0.4))
        relu_formula = axes1.get_graph_label(
            relu_graph, Tex("max(0,z)").scale(1.3),direction=RIGHT).shift(2.5*DOWN+RIGHT)
        relu_title = TexText('Rectified Linear Unit').to_edge(UP).shift(DOWN)
        relu_description = TexText('Simple, Effective, Less Computation').next_to(axes1,DOWN)
        self.fade_all_except_title()
        self.play(Write(axes1, lag_ratio=0.01, run_time=1))
        self.play(
            ShowCreation(relu_graph),
            Write(relu_title),
            # Write(lbl),
            # Write(tip),
        )
        self.wait()
        self.play(FadeIn(relu_label, RIGHT))
        self.wait(2)
        self.play(Write(relu_formula))
        self.play(Indicate(relu_formula[0][0:3]))
        self.play(Indicate(relu_formula[0][4],scale_factor=1.5))
        self.play(Indicate(relu_formula[0][6]),scale_factor=1.5)
        self.wait(2)
        self.play(Indicate(relu_formula[0][6]), scale_factor=1.5)
        self.wait(2.5)
        self.play(Indicate(relu_formula[0][4], scale_factor=1.5))
        self.wait(7)
        self.play(Write(relu_description),run_time=3)
        self.wait(2)
        tanh_label = axes2.get_graph_label(
            tanh_graph, Text("Tanh Function").scale(0.4))
        tanh_formula = axes2.get_graph_label(
            tanh_graph, Tex("\\frac{(e^z-e^{-z})}{(e^z+e^{-z})}").scale(0.8), direction=RIGHT).shift(1.5*DOWN+0.2*RIGHT)
        tanh_title = TexText('Hyperbolic Tangent Function').to_edge(UP).shift(DOWN)
        tanh_description = TexText(
            'Used in Binary Classification\\\Performs better than Sigmoid Function').next_to(self.grp, DOWN).scale(0.8)
        # self.embed()
        NeuralNetwork.arguments['layer_sizes'] = [4, 6, 6, 4]
        NeuralNetwork.add_neurons(self, False)
        NeuralNetwork.add_edges(self, False)
        NeuralNetwork.group1(self)
        self.grp.shift(3*RIGHT)
        relu = VGroup(axes1,relu_graph,relu_formula)
        tanh = VGroup(axes2,tanh_graph,tanh_formula)
        # self.embed()
        self.play(ReplacementTransform(relu,tanh),FadeOut(VGroup(relu_description,relu_title,relu_label)))
        self.play(Write(tanh_title))
        self.wait()
        self.play(Write(tanh_label))
        self.wait(4)
        one = axes2.y_axis.get_number_mobject(1,font_size=24)
        negative_one = axes2.y_axis.get_number_mobject(-1,font_size=24)
        self.play(Write(negative_one))
        self.play(Write(one))
        self.wait(2)
        dot = Dot(color=BLUE)
        dot.move_to(axes2.i2gp(0, tanh_graph))
        self.play(GrowFromPoint(dot,dot.get_center(), scale=0.5))

        x_tracker = ValueTracker(0)
        f_always(
            dot.move_to,
            lambda: axes2.i2gp(x_tracker.get_value(), tanh_graph)
        )

        self.play(x_tracker.animate.set_value(4), run_time=3)
        self.play(x_tracker.animate.set_value(-4), run_time=5)
        self.wait(2)
        # tanh_grp = VGroup(axes2, tanh_graph, tanh_label,
        #                   tanh_formula, tanh_title)
        tanh.add(tanh_label)
        tanh_grp_copy = tanh.copy().shift(4*LEFT).scale(0.8)
        self.play(Transform(tanh,tanh_grp_copy),FadeOut(VGroup(one,negative_one,dot)))
        self.remove(x_tracker)
        self.play(Write(self.grp))
        self.play(Write(tanh_description))
        self.wait()
        self.play(Indicate(self.grp.layers[3]))

        self.wait(5)
        self.fade_all_except_title()
        NeuralNetwork.arguments['layer_sizes'] = [4, 6, 6, 4]
        NeuralNetwork.add_neurons(self, False)
        NeuralNetwork.add_edges(self, False)
        NeuralNetwork.group1(self)
        self.grp.scale(1.2).to_edge(RIGHT).shift(LEFT)
        question = TexText('Which should you use?').move_to([3,3,0])
        # self.add(question)
        softmax = TexText('Softmax Function').to_edge(UP).shift(DOWN)
        softmax_formula = Tex('\\text{Softmax}(x)=\\frac{e^z}{\sum e^z}')
        self.play(Write(softmax))
        self.play(Write(softmax_formula))
        self.play(softmax_formula.animate.shift(3.5*LEFT))
        self.play(Write(self.grp))

        softmax_group = self.label('a',VGroup(),3)
        self.play(Write(softmax_group))
        softmax_sum = Tex('\sum_{a=0} = 1').next_to(self.grp.layers[2],UP)
        softmax_group_copy = softmax_group.copy()
        self.wait(5)
        self.play(Transform(softmax_group_copy,softmax_sum))
        softmax_application = TexText('Used in Multiclass Classification').next_to(self.grp,DOWN)
        self.wait(2)
        ans_layer = VGroup(Tex('0.1').scale(0.5).move_to(self.grp.layers[3][0][0]),Tex('0.5').scale(0.5).move_to(self.grp.layers[3][0][1]),Tex('0.2').scale(0.5).move_to(self.grp.layers[3][0][2]),Tex('0.1').scale(0.5).move_to(self.grp.layers[3][0][3])).set_color(BLUE)
        self.play(*[Transform(softmax_group[i],ans_layer[i]) for i in range(len(ans_layer))])
        self.wait(5)
        self.play(Write(softmax_application))
        self.wait(6)
        self.fade_all_except_title()
        self.play(Transform(self.title,TexText('How to choose Activation Function').to_edge(UP+LEFT)))
        Choose.construct(self)
        self.fade_all_except_title()
        self.play(Transform(self.title,TexText('Bias: Further Explanation').to_edge(UP+LEFT)))
        Bias.construct(self)
        self.fade_all_except_title()
        EndScreen.construct(self)
        self.embed()

    def first_part(self):
        #asset, first part
        self.title = TexText('What is Neural Network?').to_corner(UP+LEFT)
        brain = SVGMobject('assets/brain.svg')
        brain.flip(LEFT)
        algorithm = TexText('Algorithm').next_to(brain,LEFT)
        question = Tex('?').next_to(brain,RIGHT)
        ans = TexText("It's an orange.").next_to(self.grp.layers[3][0][0],RIGHT)
        question2nd = TexText('What is a neuron?').to_corner(UP+LEFT)
        arrow = Arrow([-2,2.5,0],[2,2.5,0])
        self.play(Write(self.title))
        self.wait()
        self.play(Write(VGroup(algorithm)))
        self.play(DrawBorderThenFill(brain),Write(question))
        self.wait()
        self.play(VGroup(algorithm,question,brain).animate.to_edge(LEFT))
        self.wait(2)
        self.subset()
        # self.embed()
        # self.wait()
        self.fade_all_except_title()

        x = [[random.uniform(0, 1) for x1 in range(6)], [
                random.uniform(0, 1) for x1 in range(6)]]

        org = SVGMobject('assets/orange2.0.svg')
        org.set_color(ORANGE).to_edge(LEFT).flip(LEFT)

        self.hid_layer = Brace(VGroup(self.grp.layers[1],self.grp.layers[2]),UP)
        self.in_layer_text = TexText('Input Layer').next_to(self.grp.layers[0],1.5*DOWN)
        self.hid_layer_text = self.hid_layer.get_text('Hidden Layer').scale(0.8)
        self.out_layer_text = TexText('Output Layer').next_to(self.grp.layers[3],1.5*DOWN)
        t = Tex('248').move_to(self.grp.layers[0][0][0]).scale(0.5)
        t1 = Tex('148').move_to(self.grp.layers[0][0][1]).scale(0.5)
        t2 = Tex('6').move_to(self.grp.layers[0][0][2]).scale(0.5)
        t3 = Tex('1').move_to(self.grp.layers[0][0][3]).scale(0.5)
        self.texts = VGroup(t,t1,t2,t3)
        self.add(self.grp)
        org_transform = org.copy()
        self.play(DrawBorderThenFill(org))
        self.play(ReplacementTransform(org_transform,self.texts))
        self.wait()
        self.play(ShowCreation(arrow))
        self.play(*[ApplyMethod(self.grp.layers[1][0][x1].set_fill,
                                WHITE, float(x[0][x1])) for x1 in range(6)])
        self.play(*[ApplyMethod(self.grp.layers[2][0][x2].set_fill,
                                WHITE, float(x[1][x2])) for x2 in range(6)])
        self.wait(4)
        self.play(ApplyMethod(self.grp.layers[3][0][0].set_fill,WHITE,1))
        self.wait()
        self.play(Write(ans))
        self.play(FadeOut(org),FadeOut(arrow),FadeOut(ans),FadeOut(self.texts))
        self.wait(3)

        #Layer
        shade_area = Rectangle(height=FRAME_HEIGHT,width=FRAME_WIDTH-6).to_edge(RIGHT)
        shade = BackgroundRectangle(shade_area)
        shade_area_hid1 = Rectangle(height=FRAME_HEIGHT,width=5).to_edge(LEFT)
        shade_area_hid2 = Rectangle(height=FRAME_HEIGHT,width=3.5).to_edge(RIGHT)
        shade_area_out = Rectangle(height=FRAME_HEIGHT,width=FRAME_WIDTH-6).to_edge(LEFT)
        shade_hid1 = BackgroundRectangle(shade_area_hid1)
        shade_hid2 = BackgroundRectangle(shade_area_hid2)
        shade_out = BackgroundRectangle(shade_area_out)
        org_transform = org.copy()

        self.play(Write(self.in_layer_text),ShowCreation(shade))
        self.wait(4)
        self.play(DrawBorderThenFill(org))
        self.bring_to_back(org)
        self.play(ReplacementTransform(org_transform,self.texts))
        self.play(FadeOut(self.texts))
        self.wait(3)
        self.play(ShowCreationThenDestruction(self.edges[0]))
        self.wait(2)
        self.play( GrowFromCenter(self.hid_layer), Write(self.hid_layer_text),ReplacementTransform(shade.copy(),shade_hid2),ReplacementTransform(shade,shade_hid1))
        self.bring_to_back(self.hid_layer)
        self.bring_to_back(self.hid_layer_text)
        self.wait(2)
        self.play(ShowCreationThenDestruction(self.edges[1]))
        self.wait(3)
        self.play(ShowCreationThenDestruction(self.edges[2]))
        self.wait(2)
        shade_out_copy = shade_out.copy()
        self.play(Write(
            self.out_layer_text), ReplacementTransform(shade_hid2, shade_out),ReplacementTransform(shade_hid1, shade_out_copy))
        self.play(Write(ans),FadeOut(shade_out_copy))
        self.wait(2)
        grp=self.grp
        NeuralNetwork.arguments['layer_sizes'] = [4, 6, 6,6,6, 1]
        NeuralNetwork.add_neurons(self, False)
        NeuralNetwork.add_edges(self, False)
        NeuralNetwork.group1(self)
        self.grp.scale(0.7).to_corner(UP+RIGHT)

        # # #explanation: speech recongition example
        self.remove(self.in_layer_text,org,self.hid_layer_text,self.hid_layer)
        self.play(FadeOut(VGroup(self.out_layer_text, shade_out, ans)))
        grp_fade = VGroup()
        print('---')
        for mob in self.mobjects:
            grp_fade.add(mob) if mob is not self.title else None
            print(mob)
        # self.embed()
        # self.fade_all_except_grp(grp)
        # self.add(grp)
        self.wait()
        self.play(ReplacementTransform(grp, self.grp))

        explanation = VGroup()
        soundwave = SVGMobject('assets/soundwavee.svg').scale(0.5).to_edge(LEFT)
        arrow = Arrow([0,0,0],[1,0,0]).next_to(soundwave,RIGHT)
        explanation1 = TexText('Frequency/Pitch of Sound').next_to(arrow,RIGHT)
        arrow1 = Arrow([0, 0, 0], [1, 0, 0]).next_to(explanation1, RIGHT)
        explanation2 = TexText('Phonemes').next_to(arrow1,RIGHT)
        explanation20 = TexText('Code').next_to(explanation2,UP)
        explanation21 = TexText('C o d e').next_to(explanation2,UP)
        arrow2 = Arrow([0,0,0],[0,-1,0]).next_to(explanation2,DOWN)
        explanation3 = TexText('Word').next_to(arrow2,DOWN)
        arrow3 = Arrow([0, 0, 0], [-1, 0, 0]).next_to(explanation3, LEFT)
        explanation4 = TexText('Phrases').next_to(arrow3,LEFT)
        arrow4 = Arrow([0, 0, 0], [-1, 0, 0]).next_to(explanation4, LEFT)
        output = TexText('Hello World').next_to(arrow4,LEFT)
        explanation.add(arrow,explanation1,arrow1,explanation2,arrow2,explanation3,arrow3,explanation4,arrow4)
        self.wait(3)
        self.play(DrawBorderThenFill(soundwave),Indicate(self.grp.layers[0],color=YELLOW, run_time=2))
        self.wait(3)
        self.play(Write(VGroup(arrow,explanation1)),Indicate(self.grp.layers[1],color=YELLOW, run_time=2))
        self.wait(3)
        self.play(Write(VGroup(arrow1,explanation2)),Indicate(self.grp.layers[2],color=YELLOW, run_time=2))
        self.wait(9)
        self.play(Write(explanation20))
        self.wait(3)
        self.play(TransformMatchingTex(explanation20,explanation21,path_arc = PI/2))
        self.wait(4)
        self.play(Write(VGroup(arrow2, explanation3)), Indicate(
            self.grp.layers[3], color=YELLOW, run_time=2))
        self.wait(2)
        self.play(Write(VGroup(arrow3, explanation4)),Indicate(self.grp.layers[4],color=YELLOW, run_time=2))
        self.wait(3.5)
        self.play(Write(VGroup(arrow4,output)),Indicate(self.grp.layers[5],color=YELLOW, run_time=2))
        self.wait(8)
        hp = TexText('Based on Human Perspective').to_edge(LEFT).shift(2*UP).to_edge(LEFT)
        not_exactly = TexText("Not Exactly What's Happening").next_to(hp,DOWN).to_edge(LEFT)
        self.play(Write(hp))
        self.wait(9)
        self.play(Write(not_exactly))
        self.wait(3)
        self.play(FadeOut(VGroup(explanation,output,explanation21)))
        self.play(ReplacementTransform(soundwave.copy(),explanation),run_time=2)
        self.wait(2)
        self.play(Indicate(self.grp.layers[1]),run_time=0.5)
        self.play(Indicate(self.grp.layers[2]),run_time=0.5)
        self.play(Indicate(self.grp.layers[3]),run_time=0.5)
        self.play(Indicate(self.grp.layers[4]), run_time=0.5)
        self.play(ReplacementTransform(explanation.copy(),output))
        self.play(Indicate(self.grp.layers[5], color=YELLOW, run_time=2))
        self.wait(1.5)

        grp = self.grp
        NeuralNetwork.arguments['layer_sizes'] = [4, 6, 6, 2]
        NeuralNetwork.add_neurons(self, False)
        NeuralNetwork.add_edges(self, False)
        NeuralNetwork.group1(self)
        self.grp.scale(1.5)
        self.fade_all_except_grp(grp)
        self.play(ReplacementTransform(grp, self.grp))

        # self.embed()
        self.wait()
        self.play(Transform(self.title, question2nd))
        self.wait(12)
        self.decimal = DecimalNumber(
            0, num_decimal_places=3, include_sign=False, unit=None)
        self.decimal.add_updater(lambda d: d.set_value(
            float(self.grp.layers[1][0][3].get_fill_opacities()[0])))
        self.decimal.scale(0.2).move_to(
            self.grp.layers[1][0][3].set_fill(WHITE, 0))
        self.play(self.camera.frame.scale, 0.1,
                  self.camera.frame.move_to, self.grp.layers[1][0][3])
        self.play(FadeOut(self.grp.edges))
        self.bring_to_front(self.decimal)
        self.play(ApplyMethod(self.grp.layers[1][0][3].set_opacity, 0))
        self.wait()
        self.play(ApplyMethod(self.grp.layers[1][0][3].set_opacity, 1))
        self.wait()
        self.play(ApplyMethod(self.grp.layers[1][0][3].set_opacity, 0))
        self.wait()
        self.play(ApplyMethod(self.grp.layers[1][0][3].set_opacity, 1))
        self.wait()




    def shallow_nn_representation(self):
        NeuralNetwork.arguments['layer_sizes'] = [3,1,1]
        NeuralNetwork.add_neurons(self, False)
        NeuralNetwork.add_edges(self, False)
        NeuralNetwork.group1(self)
        l = "x"

        self.grp.scale(3)
        self.grp.layers[1][0][0].scale(2)
        input_group = VGroup()

        for x in range(NeuralNetwork.arguments['layer_sizes'][0]):
            label = Tex(f"{l}_"+"{"+f"{x+1}"+"}")
            label.move_to(self.grp.layers[0][0][x])
            input_group.add(label)
        y_hat = Tex('\hat{y}')
        y_hat.move_to(self.grp.layers[2][0][0])
        sigma = Tex('\sigma').shift(0.5*DOWN)
        line = Line(self.grp.edges[0][1],self.grp.edges[1][0])
        log_reg_formula = Tex('w^Tx+b').scale(0.8).shift(0.4*UP)
        self.play(Write(self.grp),Write(input_group), Write(
            log_reg_formula), Write(line), Write(sigma),Write(y_hat))
        return

    def weights(self):
        NeuralNetwork.arguments['layer_sizes'] = [3, 1, 1]
        NeuralNetwork.add_neurons(self, False)
        NeuralNetwork.add_edges(self, False, False)
        NeuralNetwork.group1(self)
        l = "x"
        weight = TexText('What is weight?').to_corner(UP+LEFT)
        self.grp.scale(3)
        self.grp.layers[1][0][0].scale(2)
        input_group = VGroup()
        input_label = TexText('Input').next_to(self.grp.layers[0][0][0], UP)
        y_hat_label = TexText('Predicted Output').next_to(
            self.grp.layers[2][0][0], UP)
        for x in range(NeuralNetwork.arguments['layer_sizes'][0]):
            label = Tex(f"{l}_"+"{"+f"{x+1}"+"}")
            label.move_to(self.grp.layers[0][0][x])
            input_group.add(label)
        y_hat = Tex('\hat{y}')
        y_hat.move_to(self.grp.layers[2][0][0])
        shallow_nn = VGroup(self.grp, input_group, y_hat)
        self.play(Write(shallow_nn),
                    Write(input_label), Write(y_hat_label),
                    Transform(self.title,weight))
        self.bring_to_back(self.grp.layers[0])
        self.play(ZoomInThenZoomOut(self.grp.layers[0], color=WHITE),ZoomInThenZoomOut(input_group))
        self.wait()
        self.play(ZoomInThenZoomOut(self.grp.layers[1],color=WHITE))
        dashed_line_group = VGroup(*[DashedLine(self.grp.edges[0][x].get_center(), [0, 2.5, 0]) for x in range(
            NeuralNetwork.arguments['layer_sizes'][0])], VGroup(DashedLine(self.grp.edges[1][0].get_center(), [0, 2.5, 0])))
        w_layer_group = VGroup()
        l = "w"
        synapse = TexText('Edges (Connections)').next_to(
            dashed_line_group[0].get_end(), UP)
        for x in range(NeuralNetwork.arguments['layer_sizes'][0]):
            label = Tex(f"{l}_"+"{"+f"{x+1}"+"}")
            label.move_to(self.grp.edges[0][x]).shift(0.4*UP)
            w_layer_group.add(label)
        self.wait()
        self.play(Write(dashed_line_group))
        self.play(Write(synapse))
        self.wait()
        self.play(FadeOut(synapse), FadeOut(dashed_line_group))
        self.wait()
        self.play(Write(w_layer_group))
        self.wait(1.5)
        self.play(ZoomInThenZoomOut(w_layer_group[0]))
        self.wait(4)
        self.play(ZoomInThenZoomOut(self.grp.edges[0]))
        self.wait(7)
        arrow = Arrow([0, 0, 0], [0, 1, 0]).next_to(w_layer_group[0], RIGHT)
        arrow3 = Arrow([0, 1, 0], [0, 0, 0]).next_to(w_layer_group[0], RIGHT)

        # zero = Tex('0').next_to(w_layer_group[0], RIGHT)
        v = ValueTracker(0)
        x1 = DecimalNumber(0, num_decimal_places=1, include_sign=False, unit=None).move_to(
            self.layers[0][0][0].get_center()).scale(0.5)
        z = DecimalNumber(0, num_decimal_places=2,
                          include_sign=False, unit=None)
        x1.add_updater(lambda m: m.set_value(v.get_value()))
        z.add_updater(lambda d: d.set_value(
            float(self.grp.layers[1][0][0].get_fill_opacities()[0])))

        self.grp.layers[1][0][0].set_fill(WHITE, 0)
        self.play(ZoomInThenZoomOut(self.grp.layers[0][0][0]))
        self.bring_to_back(self.grp.layers[0])
        self.wait()
        self.play(ZoomInThenZoomOut(self.grp.layers[1][0][0],color=WHITE))
        self.wait(2)
        self.play(Write(arrow), ReplacementTransform(input_group[0], x1))
        self.add(z)

        self.play(v.animate.set_value(7),
                  ApplyMethod(self.grp.layers[1][0][0].set_fill, WHITE, 0.8), run_time=2)
        self.wait(2)
        self.grp.layers[1][0][0].set_fill(WHITE, 0)
        v.set_value(0)
        self.play(Transform(arrow, arrow3))
        self.play(v.animate.set_value(7),
                  ApplyMethod(self.grp.layers[1][0][0].set_fill, WHITE, 0.4), run_time=2)
        self.wait(3)
        self.remove(x1, z,v)
        self.play(FadeOut(VGroup(self.grp, arrow, w_layer_group, input_group,input_label,y_hat_label,y_hat)))
        # self.play(ReplacementTransform(arrow,zero),FadeOut(arrow1))
        # self.grp.layers[1][0][0].set_fill(WHITE,0)
        # self.play(ShowCreationThenFadeOut(arrow2),run_time=0.5)
        # arrow2.flip(LEFT)
        # self.play(ShowCreationThenFadeOut(arrow2),run_time=0.5)
        # self.wait()

        # self.play(ReplacementTransform(zero,arrow3))
        # self.play(ApplyMethod(self.grp.layers[1][0][0].set_fill, WHITE, 0.2))

    def nn_representation(self):
        return

    def label(self,l,group,layer=0,edges=False,bias=False):
        if bias is True:
            for x in range(len(NeuralNetwork.arguments['layer_sizes'])-1):
                label = Tex(f"{l}_"+"{"+f"{x}"+"}").scale(0.8)
                label.move_to(self.bias_layer[x])
                group.add(label)
        else:
            for x in range(NeuralNetwork.arguments['layer_sizes'][layer]):
                label = Tex(f"{l}_"+"{"+f"{x}"+"}").scale(0.8)

                label.move_to(self.grp.edges[layer][x]).shift(
                    0.4*UP) if edges is True else label.move_to(self.grp.layers[layer][0][x])
                group.add(label)

        return group

    def connections(self,x,y,z,wait=1,highlight=False):
        i=0
        while i != NeuralNetwork.arguments['layer_sizes'][0]:
            self.edge_transform_1.add(self.edges[0][x])
            print(0,x)
            i+=1
            x+=NeuralNetwork.arguments['layer_sizes'][1]


        self.play(ShowCreationThenDestruction(self.edge_transform_1))
        self.wait(wait)
        if highlight is True:
            self.play(Indicate(self.grp.layers[1][0][0],color=WHITE))
            self.wait(2)
        self.play(*[ShowCreationThenDestruction(self.edges[1][i]) for i in range(y,z)])

    def dot(self):
        dot_layer = VGroup(*[Dot().move_to(self.grp.layers[0][0][i].get_center()) for i in range(NeuralNetwork.arguments['layer_sizes'][0])])
        dot_layer.set_fill(WHITE,0)
        return dot_layer

    def fade_all_except_title(self):
        grp_fade = VGroup()
        print('---')
        for mob in self.mobjects:
            grp_fade.add(mob) if mob is not self.title else None
            print(mob)
        self.play(
            FadeOut(grp_fade)
        )

    def subset(self):
        nn = TexText('Neural Network')
        dl = TexText('Deep Learning').shift(1.3*UP)
        ml = TexText('Machine Learning').shift(2.3*UP)
        c = Circle(stroke_color=RED).stretch(2, dim=0)
        c1 = Circle(radius=1.5, stroke_color=GREEN).stretch(
            2, dim=0).shift(0.5*UP)
        c2 = Circle(radius=2, stroke_color=BLUE).stretch(2, dim=0).shift(UP)
        subset = VGroup(nn,dl,ml,c,c1,c2).scale(0.8).to_edge(RIGHT).shift(DOWN)
        self.play(Write(VGroup(c, nn)))
        self.wait()
        self.play(Write(VGroup(c1, dl)))
        self.wait()
        self.play(Write(VGroup(c2, ml)))

    def fade_all_except_grp(self,grp):
        grp_fade = VGroup()
        for mob in self.mobjects:
            grp_fade.add(mob) if mob is not grp else None
        self.play(
            FadeOut(grp_fade)
        )

# color detection
# edge detection
# group detection
#TODO
