"""ReLU Activation in a Pair of Neurons .PAG 83

    El bias es donde empieza la funcion y el peso
    es la inlinacion

    input 0-------->O(relu) ---> ____|____ X = 0 ; y= 0
    bias  0---------^ 
    
    input 1-------->O(relu) ---> _____|/   X = 1 ; y= 1
    bias  0---------^  

    input 1-------->O(relu) ---> ___/_|__ X = -0.5 ; y>=0 = 1
    bias  -0.5------^ 

    input -1------->O(relu) ---> _____|_\_ X = -0.5 ; y>=0 = 1
    bias  0.5------^ 

    Pag 83 graficas

    interaccion de dos neurones con ejemplos basicos donde la salida
    es igual a la entrada, esto esporque el peso del segundo neuron 
    es igual a 1 y el bias es cero lo cual produce nada de offset

    interaccion de dos neurones con ejemplos basicos donde la salida
    es diferente a al entrada, es porque el peso del segundo neuron
    es diferente de 1 y el bias es diferente de ceo, lo cual produce
    offset

    pag 83
    """