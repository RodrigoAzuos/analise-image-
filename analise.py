import cv2
import imutils
import numpy as np

def show_image(titulo, imagem):
	cv2.imshow(titulo, imagem)
	cv2.waitKey(0)
	cv2.imwrite(titulo+".jpg", imagem)

def main():

	imagem = cv2.imread('dados.jpg')

	imagem_cinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
	show_image("Rodrigo_imagem_cinza",imagem_cinza)

	imagem_suave = cv2.blur(imagem_cinza, (7,7))
	show_image("Rodrigo_imagem_suave",imagem_suave)

	imagem_bin = cv2.threshold(imagem_suave, 110, 250, cv2.THRESH_BINARY_INV)[1]
	show_image("Rodrigo_imagem_binaria",imagem_bin)

	imagem_borda = cv2.Canny(imagem_bin, 70,150)
	show_image("Rodrigo_imagem_borda",imagem_borda)

	quant_obj = cv2.findContours(imagem_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	quant_obj = imutils.grab_contours(quant_obj)
	imagem_final = imagem.copy()

	for n in quant_obj:
	    cv2.drawContours(imagem_final, [n], -1, (85,200,100), 3)

	resultado = "{} dados!".format(len(quant_obj))
	cv2.putText(imagem_final, resultado, (10,25), cv2.FONT_HERSHEY_SIMPLEX,
	            0.7, (85,200,100),2)

	show_image("Rodrigo_imagem_final",imagem_final)

if __name__ == '__main__':
	main()

