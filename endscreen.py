from manimlib import *
from nn import *
# from backward_teaser import *
from graph_scene import *
from bias_update import *
class EndScreen(Bias,graph_scene):
    def construct(self):
        self.title =  TexText('Thanks for watching!').move_to([-4,3,0])
        NeuralNetwork.arguments['layer_sizes'] = [4, 6, 6,1]
        NeuralNetwork.add_neurons(self, False)
        NeuralNetwork.add_edges(self, False)
        NeuralNetwork.group1(self)
        self.grp.scale(1.2)
        subscribe = TexText('SUBSCRIBE!').scale(0.8)
        subs_button = RoundedRectangle(height=1,width=3).set_fill(RED,1).set_stroke(RED,1)
        like = SVGMobject('assets/like.svg').scale(0.3).next_to(subs_button,LEFT)
        like.flip(LEFT)
        subs = VGroup(subs_button,subscribe,like).scale(0.8)
        subs.to_edge(LEFT)
        subscribed = TexText('SUBSCRIBED').scale(0.8)
        subbed_button = RoundedRectangle(
            height=1, width=3).set_fill(GREY, 1).set_stroke(GREY, 1)
        liked = SVGMobject(
            'assets/liked.svg').scale(0.3).next_to(subbed_button,LEFT).set_color(BLUE)
        liked.flip(LEFT)
        subbed = VGroup(subbed_button, subscribed,liked).scale(0.8).to_edge(RIGHT)

        x = [[random.uniform(0, 1) for x1 in range(6)], [
                random.uniform(0, 1) for x1 in range(6)]]
        self.play(Transform(self.title,TexText('Thanks for watching!').to_edge(UP+LEFT)))
        self.wait(7)
        comments = TexText('Let me know in the comment section!')
        self.play(Write(comments))
        self.play(FadeOut(comments))
        self.play(GrowFromCenter(self.grp))
        self.play(Write(subs))
        dot = self.grp.layers[0].copy()
        # dot.set_stroke(BLACK,1)
        S = TexText('S').move_to(self.grp.layers[0][0][0]).scale(0.5)
        U = TexText('U').move_to(self.grp.layers[0][0][1]).scale(0.5)
        B = TexText('B').move_to(self.grp.layers[0][0][2]).scale(0.5)
        S_END = TexText('S').move_to(self.grp.layers[0][0][3]).scale(0.5)
        subs_word = VGroup(S,U,B,S_END)
        self.play(Transform(subs,subs_word))
        self.play(*[ApplyMethod(self.grp.layers[1][0][x1].set_fill,WHITE, float(x[0][x1])) for x1 in range(6)], *[ApplyMethod(self.grp.layers[2][0][x1].set_fill, WHITE, float(x[1][x1])) for x1 in range(6)])
        self.grp.layers[3][0].animate.set_fill(WHITE,1)
        self.play(ApplyMethod(self.grp.layers[3][0][0].set_fill,WHITE,1))
        self.play(Write(subbed))
        self.wait(2)
        grp_fade = VGroup()
        for mob in self.mobjects:
            grp_fade.add(mob) if mob is not self.title else None
        self.play(
            FadeOut(grp_fade)
        )
        subscribe = TexText('Subscribe!').move_to([-4,-1,0])
        series = TexText('Neural Network\\\ Crash Course Series').move_to([3,-1.5,0])
        self.play(Write(VGroup(subscribe,series)))
        self.wait(13)





        # self.embed()
        # self.wait(30)

    def dot(self):
        dot_layer = VGroup(*[Dot().move_to(self.grp.layers[0][0][i].get_center())
                           for i in range(NeuralNetwork.arguments['layer_sizes'][0])])
        dot_layer.set_fill(WHITE, 0)
        return dot_layer
