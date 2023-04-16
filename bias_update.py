from manimlib import *
from nn import NeuralNetwork
from applications import Application
from chooose import Choose


class Bias(Choose):
    def construct(self):
        axes = Axes((-10, 10), (0, 1.2)).scale(0.5)
        axes.add_coordinate_labels(
            x_values=[-10, -5, 0, 5, 10], y_values=[], font_size=15)
        sigmoid_w0 = axes.get_graph(
            lambda x: 1.0 / (1.0 + math.exp(-0.5*x)),
            use_smoothing=False,
            color=RED,
        )
        sigmoid_w1 = axes.get_graph(
            lambda x: 1.0 / (1.0 + math.exp(-1*x)),
            use_smoothing=False,
            color=GREEN,
        )
        sigmoid_w2 = axes.get_graph(
            lambda x: 1.0 / (1.0 + math.exp(-5*x)),
            use_smoothing=False,
            color=BLUE,
        )
        non_linear_label = Tex(
            "\\sigma (0.5*x)", color=RED).move_to([4.9, 2, 0])
        non_linear_label_1 = Tex(
            "\\sigma (1*x)", color=GREEN).move_to([4.8, 1, 0])
        non_linear_label_2 = Tex(
            "\\sigma (5*x)", color=BLUE).move_to([4.8, 0, 0])

        lbl_3 = axes.x_axis.get_number_mobject(3, font_size=15)
        lbl_negative_3 = axes.x_axis.get_number_mobject(-3, font_size=15)
        sigma = Tex('\sigma').scale(2)
        self.wait(3)
        self.play(Write(sigma))
        self.play(sigma.animate.shift(1.5*LEFT),run_time=2)
        self.play(sigma.animate.shift(3*RIGHT),run_time=2)
        self.wait(2)
        self.play(FadeOut(sigma))
        self.play(ShowCreation(axes),run_time=2)
        self.play(ShowCreation(sigmoid_w1))
        self.wait()
        self.play(Write(non_linear_label_1))
        self.wait()
        self.play(ReplacementTransform(sigmoid_w1.copy(),
                  sigmoid_w0), Write(non_linear_label),run_time=2)
        self.wait()
        self.play(ReplacementTransform(sigmoid_w0.copy(),
                  sigmoid_w2), Write(non_linear_label_2),run_time=2)
        self.wait()
        sigmoid_w2_copy = sigmoid_w2.copy()
        self.play(ReplacementTransform(sigmoid_w1.copy(),sigmoid_w2_copy,run_time=3),ZoomInThenZoomOut(non_linear_label_2))
        self.wait(5.5)
        self.play(IndicateThenFadeOut(lbl_3, scale_factor=1.5),
                  FadeOut(sigmoid_w2_copy))
        self.wait(2.5)
        self.play(IndicateThenFadeOut(lbl_negative_3, scale_factor=1.5))
        # self.play(FadeOut(lbl_3))
        self.wait()
        self.play(ReplacementTransform(sigmoid_w2, sigmoid_w1.copy()), ReplacementTransform(
            sigmoid_w0, sigmoid_w1.copy()), FadeOut(VGroup(non_linear_label, non_linear_label_2)))
        self.wait(2)

        sigmoid_graph_3 = axes.get_graph(
            lambda x: 1.0 / (1.0 + math.exp(-(-6+1*x))),
            use_smoothing=False,
            color=RED,
        )
        # v_line = always_redraw(lambda: axes.get_v_line(dot.get_bottom()))
        dot = Dot().move_to(axes.i2gp(3, sigmoid_graph_3))
        line = Line(dot, axes.c2p(3, 0))
        sigmoid_graph2 = axes.get_graph(
            lambda x: 1.0 / (1.0 + math.exp(-(6+1*x))),
            use_smoothing=False,
            color=BLUE,
        )
        dot1 = Dot().move_to(axes.i2gp(-3, sigmoid_graph2))
        line1 = axes.get_v_line(dot1.get_bottom())

        bias_label = Tex("\\sigma (1*x+(-6*1))",
                         color=RED).move_to([4.9, 2, 0])
        # bias_label_1 = Tex("\\sigma (1*x)", color=GREEN).move_to([4.8,1,0])
        bias_label_2 = Tex("\\sigma (1*x+(6*1))",
                           color=BLUE).move_to([4.8, 0, 0])

        # line = axes.get_graph(lambda x: 0.5, color=YELLOW)
        self.play(ReplacementTransform(sigmoid_w1.copy(),
                  sigmoid_graph2), Write(bias_label_2))
        self.play(GrowFromPoint(dot1, dot1.get_center()), ShowCreation(line1))
        self.play(ReplacementTransform(sigmoid_w1.copy(),
                  sigmoid_graph_3), Write(bias_label))
        self.play(ShowCreation(VGroup(dot, line)))
        # self.embed()
