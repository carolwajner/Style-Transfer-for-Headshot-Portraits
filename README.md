# Style-Transfer-for-Headshot-Portraits

pip3 install opencv-python
pip3 install dlib

python3 main.py data/input.jpg

Boas referências:

https://pyimagesearch.com/2017/04/03/facial-landmarks-dlib-opencv-python/
https://dl.acm.org/doi/pdf/10.1145/280811 página 393

Constantes:

If a is barely greater than zero, then if the distance from the line to
the pixel is zero, the strength is nearly infinite. With this value for
a, the user knows that pixels on the line will go exactly where he
wants them. Values larger than that will yield a more smooth warping, but with less precise control. The variable b determines how the
relative strength of different lines falls off with distance. If it is large,
then every pixel will be affected only by the line nearest it. If bis
zero, then each pixel will be affected by all lines equally. Values of
bin the range [0.5, 2] are the most useful. The value of pis typically
in the range [0, 1 ]; if it is zero, then all lines have the same weight,
if it is one, then longer lines have a greater relative weight than
shorter lines. 