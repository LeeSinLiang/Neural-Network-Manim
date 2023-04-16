from numpy import exp
from manimlib import *
from nn import *
from applications import *


class Choose(Application):

    def construct(self):
        NeuralNetwork.arguments['layer_sizes'] = [4, 6, 6, 4]
        NeuralNetwork.add_neurons(self, False)
        NeuralNetwork.add_edges(self, False)
        NeuralNetwork.group1(self)
        self.grp.scale(1.2).to_edge(RIGHT).shift(LEFT)
        title = TexText('Activation Function').move_to([0, 2.5, 0])
        hidden_layer = TexText('Hidden Layer').move_to([-3, 1, 0])
        output_layer = TexText('Output Layer').move_to([3, 1, 0])
        b_classification = TexText(
            'Binary\\\Classification').move_to([1.5, -1, 0])
        m_classification = TexText(
            'Multiclass\\\Classification').move_to([4.5, -1, 0])
        ReLU = TexText('ReLU\\\Activation').move_to([-3, -1, 0])
        r = Rectangle().surround(ReLU)
        sigmoid = TexText('Sigmoid/Tanh\\\Activation').move_to([1.5, -3, 0])
        softmax = TexText('Softmax\\\Activation').move_to([4.5, -3, 0])
        line = Arrow(title, hidden_layer)
        line1 = Arrow(title, output_layer)
        line2 = Arrow(output_layer, b_classification)
        line3 = Arrow(output_layer, m_classification)
        arrow = Arrow(hidden_layer, ReLU)
        arrow1 = Arrow(b_classification, sigmoid)
        arrow2 = Arrow(m_classification, softmax)
        self.wait(2)
        self.play(Write(title))
        self.wait(2)
        self.play(ShowCreation(line),  ShowCreation(line1))
        self.wait(9)
        self.play(Write(ReLU))
        self.wait()
        self.play(ShowCreation(arrow))
        self.play(Write(hidden_layer))
        self.play(Write(b_classification))
        self.wait()
        self.play(ShowCreation(arrow1))
        self.play(Write(sigmoid))
        self.wait()
        self.play(Write(output_layer))
        self.play(ShowCreation(line2))
        self.play(ShowCreation(line3))
        self.wait()
        self.play(Write(softmax))
        self.wait()
        self.play(Write(m_classification))
        self.play(ShowCreation(arrow2))
        self.wait()

    def label(self, l, group, layer=0, edges=False, bias=False):
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

#TODO softmax function
