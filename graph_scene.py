from manimlib import *

class graph_scene(Scene):
    def construct(self):

        axes = Axes((-5, 5), (0, 1),
                    y_axis_config={
            "decimal_number_config": {
                "num_decimal_places": 1,
            }
        },
            width=10).scale(0.4).shift(0.5*UP)
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
            width=10).scale(0.4).shift(0.5*DOWN)
        relu_graph = axes1.get_graph(
            lambda x: max(0, x),
            use_smoothing=False,
            color=YELLOW,
        )
        axes2 = Axes((-5, 5), (-1, 1),
                    y_axis_config={
            "decimal_number_config": {
                "num_decimal_places": 1,
            }
        },
            width=10).scale(0.4).shift(0.5*DOWN)
        tanh_graph = axes2.get_graph(
            lambda x: (math.exp(x) - math.exp(-x)) /
            (math.exp(x) + math.exp(-x)),
            use_smoothing=False,
            color=YELLOW,
        )

        sigmoid_name = TexText('Sigmoid Function').next_to(axes,DOWN)
        relu_name = TexText('ReLU Function').next_to(axes1,DOWN)
        tanh_name = TexText('Tanh Function').next_to(axes2,DOWN)
        softmax_formula = Tex('\\text{Softmax}(x)=\\frac{e^z}{\sum e^z}')
        softmax_name = TexText('Softmax Function').next_to(softmax_formula,DOWN)

        sigmoid_grp = VGroup(axes,sigmoid_graph,sigmoid_name).shift(1.2*UP+3*LEFT)
        relu_grp = VGroup(axes1,relu_graph,relu_name).shift(2.2*UP+3*RIGHT)
        tanh_grp = VGroup(axes2,tanh_graph,tanh_name).shift(1.5*DOWN+3*LEFT)
        softmax_grp = VGroup(softmax_name,softmax_formula).shift(2*DOWN+3*RIGHT)
        self.play(Write(sigmoid_grp))
        self.wait()
        self.play(Write(relu_grp))
        self.wait()
        self.play(Write(tanh_grp))
        self.wait()
        self.play(Write(softmax_grp))
