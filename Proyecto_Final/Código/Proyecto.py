import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier

# Datos de ejemplo de usuarios
usuarios_data = {
    'id': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
    'nombre': ['Usuario1', 'Usuario2', 'Usuario3', 'Usuario4', 'Usuario5', 'usuario6', 'usuario7', 'usuario8',
               'usuario9', 'usuario10', 'usuario11', 'usuario12', 'usuario13', 'usuario14', 'usuario15'],
    'altura': [170, 165, 180, 175, 160, 185, 162, 177, 183, 190, 184, 167, 183, 171, 173],
    'peso': [76, 58, 67, 85, 60, 66, 77, 71, 64, 110, 76, 58, 84, 66, 60],
    'grasa_corporal': [24, 19, 21, 25, 18, 20, 26, 17, 21, 28, 18, 22, 25, 19, 21],
    'objetivo': ['perder peso', 'mantener salud', 'ganar músculo', 'perder peso', 'mantener salud', 'ganar músculo',
                 'perder peso', 'mantener salud', 'ganar músculo', 'perder peso', 'mantener salud', 'ganar músculo',
                 'perder peso', 'mantener salud', 'ganar músculo']
}

usuarios_df = pd.DataFrame(usuarios_data)

# Datos de ejemplo de recetas
recetas_data = {
    'id': [1, 2, 3, 4, 5, 6, 7, 8, 9],
    'nombre': ['Batido de Proteína y Plátano', 'Wrap de Pollo con Aguacate y Espinacas',
               'Filete de Res con Puré de Papas y Ensalada',
               'Avena con Frutas', 'Sopa de Pollo y Verduras', 'Pechuga de Pollo al Limón con Brócoli',
               'Tostadas de Aguacate con Huevo', 'Salmón al Horno con Batata y Espárragos',
               'Hamburguesa de Pavo con Patatas Fritas'],
    'ingredientes': [
        '1 plátano, 1 scoop de proteína en polvo, 1 taza de leche (puede ser de almendra, soja, etc.), 1 cucharada de mantequilla de almendra, Hielo al gusto',
        '1 tortilla integral, 250g de pechuga de pollo cocida y en tiras, 1/2 aguacate, en rodajas ,1 taza de espinacas frescas, 1 cucharada de mayonesa light, Sal y pimienta al gusto',
        '250g de filete de res, 2 papas medianas, 1 cucharada de mantequilla, 1/4 taza de leche, 1 taza de mezcla de ensaladas (lechuga, tomate, pepino), 1 cucharada de aceite de oliva, Sal y pimienta al gusto',
        '1/2 taza de avena, 1 taza de agua o leche de almendra, 1/2 taza de fresas picadas, 1 cucharada de semillas de chía',
        '250g de pechuga de pollo, en cubos, 2 tazas de caldo de pollo bajo en sodio, 1 zanahoria, en rodajas, 1 apio, en rodajas, 1/2 calabacín, en rodajas, 1/2 cebolla, picada, 1 diente de ajo, picado, Sal y pimienta al gusto',
        '250g de pechuga de pollo, 1 taza de brócoli, 1 cucharada de aceite de oliva, Jugo de 1 limón, Sal y pimienta al gusto',
        '2 rebanadas de pan integral, 1 aguacate maduro, 2 huevos, Sal y pimienta al gusto, Jugo de 1/2 limón',
        '250g de filete de salmón, 1 batata, 1 taza de espárragos, 1 cucharada de aceite de oliva, Sal, pimienta y limón al gusto',
        '150g de carne de pavo molida, 1 pan integral para hamburguesa, 1 hoja de lechuga, 1 rodaja de tomate, 1/2 aguacate, en rodajas, 1 Patata, en rodajas, 1 cucharada de aceite de oliva, Sal y pimienta al gusto'],
    'categoria': ['ganar músculo', 'ganar músculo', 'ganar músculo', 'perder peso', 'perder peso', 'perder peso',
                  'mantener salud', 'mantener salud', 'mantener salud'],
    'horario': ['desayuno', 'almuerzo', 'cena', 'desayuno', 'almuerzo', 'cena', 'desayuno', 'almuerzo', 'cena'],
    'preparacion': [
        'Mezcla todos los ingredientes en una licuadora. Licúa hasta obtener una mezcla homogénea. Sirve inmediatamente.',
        'Calienta la tortilla en una sartén. Unta la mayonesa en la tortilla. Añade el pollo, el aguacate y las espinacas. Enrolla la tortilla y corta por la mitad.',
        'Cocina el filete de res en una sartén a fuego medio-alto hasta el punto deseado. Pela y corta las papas, hiérvelas hasta que estén tiernas. Machaca las papas con la mantequilla y la leche hasta obtener un puré. Prepara la ensalada y aliña con aceite de oliva, sal y pimienta. Sirve el filete con el puré de papas y la ensalada.',
        'Cocina la avena con el agua o leche según las instrucciones del paquete. Una vez cocida, añade las fresas y las semillas de chía. Mezcla bien y sirve.',
        'En una olla grande, calienta el caldo de pollo. Añade el pollo, las verduras y el ajo. Cocina a fuego medio hasta que el pollo y las verduras estén tiernos. Añade sal y pimienta al gusto antes de servir.',
        'Sazona la pechuga de pollo con sal, pimienta y el jugo de limón. Cocina el pollo en una sartén con aceite de oliva hasta que esté bien cocido. Hierve el brócoli hasta que esté tierno. Sirve el pollo con el brócoli.',
        'Tosta el pan. Machaca el aguacate y mézclalo con sal, pimienta y jugo de limón. Extiende la mezcla de aguacate sobre las tostadas. Fríe los huevos al gusto y colócalos sobre las tostadas.',
        'Precalienta el horno a 200°C. Sazona el salmón con sal, pimienta y limón. Pela y corta la batata en rodajas. Coloca las rodajas y los espárragos en una bandeja para hornear. Rocía con aceite de oliva y hornea todo durante 20-25 minutos.',
        'Forma una hamburguesa con la carne de pavo, sazona con sal y pimienta. Cocina la hamburguesa en una sartén hasta que esté bien cocida. Asa las rodajas de batata en una bandeja para hornear con aceite de oliva, sal y pimienta a 200°C durante 20-25 minutos. Coloca la hamburguesa en el pan integral con lechuga, tomate y aguacate. Sirve con las batatas fritas.']
}

recetas_df = pd.DataFrame(recetas_data)

# Datos de ejemplo de ejercicios
ejercicios_data = {
    'id': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
    'nombre': ['Peso muerto', 'Correr o trotar', 'Saltar la cuerda', 'Sentadillas con peso', 'Burpees',
               'Flexiones', 'Dominadas', 'Press de banca', 'Fondos en paralelas', 'Ciclismo',
               'Entrenamiento de Intervalos de Alta Intensidad', 'Mountain Climbers', 'Press de Banca con Mancuernas',
               'Peso Muerto Rumano', 'Flexiones Explosivas'],
    'categoria': ['perder peso', 'perder peso', 'perder peso', 'perder peso', 'perder peso', 'ganar músculo',
                  'ganar músculo', 'ganar músculo', 'ganar músculo', 'ganar músculo', 'mantener salud',
                  'mantener salud', 'mantener salud', 'mantener salud', 'mantener salud'],
    'descripcion': ['Levantar una barra con peso desde el suelo hasta la posición erguida',
                    'Correr o trotar a un ritmo constante',
                    'Saltar una cuerda de forma continua',
                    'Ejercicio de piernas donde se baja el cuerpo flexionando las rodillas',
                    'Ejercicio de alta intensidad que combina una sentadilla y una flexión con un salto en el aire',
                    'Ejercicio de peso corporal donde se empuja el cuerpo hacia arriba desde el suelo',
                    'Ejercicio de peso corporal donde se cuelga de una barra y se tira del cuerpo hacia arriba',
                    'Levantar una barra con peso desde el pecho hasta la extensión completa de los brazos',
                    'Ejercicio de peso corporal donde se baja y se sube el cuerpo entre dos barras paralelas',
                    'Montar en bicicleta a un ritmo constante',
                    'Alternar ráfagas cortas de ejercicio intenso con períodos de descanso activo para mejorar la condición física cardiovascular',
                    'Colócate en posición de plancha con las manos debajo de los hombros y el cuerpo en línea recta. Alterna rápidamente llevando las rodillas hacia el pecho en un movimiento de carrera. Mantén el ritmo constante y la intensidad alta durante 30-60 segundos.',
                    'Acuéstate en un banco plano con una mancuerna en cada mano a la altura de los hombros. Empuja las mancuernas hacia arriba hasta que los brazos estén completamente extendidos sobre el pecho. Baja las mancuernas de manera controlada hasta la posición inicial. Repite el movimiento manteniendo la estabilidad del núcleo y la técnica adecuada.',
                    'Sujeta una barra con un agarre pronado (palmas hacia abajo) a la altura de los muslos. Mantén las piernas casi completamente extendidas y la espalda recta. Baja la barra hacia abajo manteniendo el peso en los talones y los músculos de la espalda baja comprometidos. Sube la barra hacia arriba contrayendo los glúteos y los isquiotibiales. Repite el movimiento controlado según tu capacidad y ajusta el peso según sea necesario.',
                    'Realiza una flexión estándar, bajando el cuerpo hasta que los codos estén a 90 grados. Empuja explosivamente hacia arriba para que las manos salgan del suelo. Da palmadas en el aire antes de volver a bajar para la siguiente repetición. Mantén el núcleo comprometido y la técnica adecuada para evitar lesiones.'],
    'duracion': ['10 reps', '30 mins', '15 mins', '10 reps', '15 mins',
                 '10 reps', '10 reps', '10 reps', '10 reps', '30 mins',
                 '20 mins', '1 min', '10 reps', '10 reps', '10 reps']
}

ejercicios_df = pd.DataFrame(ejercicios_data)

# Codificar las etiquetas de los objetivos
label_encoder = LabelEncoder()
usuarios_df['objetivo_encoded'] = label_encoder.fit_transform(usuarios_df['objetivo'])

# Seleccionar características y etiquetas de los usuarios
X_usuarios = usuarios_df[['altura', 'peso', 'grasa_corporal']]
y_recetas = usuarios_df['objetivo_encoded']
y_ejercicios = usuarios_df['objetivo_encoded']

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train_recetas, y_test_recetas = train_test_split(X_usuarios, y_recetas, test_size=0.3,
                                                                    random_state=42)
X_train, X_test, y_train_ejercicios, y_test_ejercicios = train_test_split(X_usuarios, y_ejercicios, test_size=0.3,
                                                                          random_state=42)

# Crear y entrenar el modelo de árbol de decisión para recetas
tree_recetas = DecisionTreeClassifier(random_state=42)
tree_recetas.fit(X_train, y_train_recetas)

# Crear y entrenar el modelo de árbol de decisión para ejercicios
tree_ejercicios = DecisionTreeClassifier(random_state=42)
tree_ejercicios.fit(X_train, y_train_ejercicios)


# Función para hacer recomendaciones
def recomendar_usuario(altura, peso, grasa_corporal):
    # Crear un DataFrame con los datos del usuario
    usuario_df = pd.DataFrame([[altura, peso, grasa_corporal]], columns=['altura', 'peso', 'grasa_corporal'])

    # Predecir el objetivo del usuario
    objetivo_pred_recetas = tree_recetas.predict(usuario_df)[0]
    objetivo_pred_ejercicios = tree_ejercicios.predict(usuario_df)[0]

    # Decodificar la etiqueta predicha
    objetivo_decoded_recetas = label_encoder.inverse_transform([objetivo_pred_recetas])[0]
    objetivo_decoded_ejercicios = label_encoder.inverse_transform([objetivo_pred_ejercicios])[0]

    # Filtrar recetas y ejercicios basados en el objetivo predicho
    recetas_recomendadas = recetas_df[recetas_df['categoria'] == objetivo_decoded_recetas].drop(
        columns=['id', 'categoria'])
    ejercicios_recomendados = ejercicios_df[ejercicios_df['categoria'] == objetivo_decoded_ejercicios].drop(
        columns=['id', 'categoria'])

    return recetas_recomendadas, ejercicios_recomendados


# Crear la interfaz gráfica
class App:
    def _init_(self, root):
        self.root = root
        self.root.title("Recomendaciones Personalizadas")

        # Aplicar un tema
        self.style = ttk.Style()
        self.style.theme_use('clam')

        # Crear un marco para organizar los widgets
        self.frame = ttk.Frame(root, padding="10 10 10 10")
        self.frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.frame.columnconfigure(0, weight=1)
        self.frame.rowconfigure(0, weight=1)

        # Crear estilos personalizados
        self.style.configure('TLabel', font=('Arial', 12))
        self.style.configure('TButton', font=('Arial', 12))
        self.style.configure('TEntry', font=('Arial', 12), padding="5")

        # Crear los elementos de la interfaz
        self.altura_label = ttk.Label(self.frame, text="Altura (en cm):")
        self.altura_label.grid(row=0, column=0, padx=10, pady=10, sticky=tk.W)
        self.altura_entry = ttk.Entry(self.frame)
        self.altura_entry.grid(row=0, column=1, padx=10, pady=10)

        self.peso_label = ttk.Label(self.frame, text="Peso (en kg):")
        self.peso_label.grid(row=1, column=0, padx=10, pady=10, sticky=tk.W)
        self.peso_entry = ttk.Entry(self.frame)
        self.peso_entry.grid(row=1, column=1, padx=10, pady=10)

        self.grasa_label = ttk.Label(self.frame, text="Porcentaje de grasa corporal:")
        self.grasa_label.grid(row=2, column=0, padx=10, pady=10, sticky=tk.W)
        self.grasa_entry = ttk.Entry(self.frame)
        self.grasa_entry.grid(row=2, column=1, padx=10, pady=10)

        self.recomendar_button = ttk.Button(self.frame, text="Recomendar", command=self.recomendar)
        self.recomendar_button.grid(row=3, column=0, columnspan=2, padx=10, pady=10)

        self.recetas_label = ttk.Label(self.frame, text="Recetas recomendadas:")
        self.recetas_label.grid(row=4, column=0, padx=10, pady=10, sticky=tk.W)

        self.recetas_frame = ttk.Frame(self.frame)
        self.recetas_frame.grid(row=5, column=0, columnspan=2)

        self.ejercicios_label = ttk.Label(self.frame, text="Ejercicios recomendados:")
        self.ejercicios_label.grid(row=6, column=0, padx=10, pady=10, sticky=tk.W)

        self.ejercicios_frame = ttk.Frame(self.frame)
        self.ejercicios_frame.grid(row=7, column=0, columnspan=2)

        # Variables para las imágenes
        self.batido_image = None
        self.wrap_image = None
        self.filete_image = None
        self.avena_image = None
        self.sopa_image = None
        self.pechuga_image = None
        self.tostadas_image = None
        self.salmon_image = None
        self.hamburguesa_image = None

        self.peso_muerto_image = None
        self.correr_image = None
        self.saltar_cuerda_image = None
        self.sentadillas_image = None
        self.burpees_image = None
        self.flexiones_image = None
        self.dominadas_image = None
        self.press_banca_image = None
        self.fondos_image = None
        self.ciclismo_image = None
        self.hiit_image = None
        self.mountain_climbers_image = None
        self.press_mancuernas_image = None
        self.peso_muerto_rumano_image = None
        self.flexiones_explosivas_image = None

    def load_image(self, filename):
        image = Image.open(filename)
        image = image.resize((100, 100), Image.LANCZOS)
        return ImageTk.PhotoImage(image)

    def recomendar(self):
        try:
            altura = float(self.altura_entry.get())
            peso = float(self.peso_entry.get())
            grasa_corporal = float(self.grasa_entry.get())
        except ValueError:
            messagebox.showerror("Error", "Por favor, ingrese valores válidos.")
            return

        recetas, ejercicios = recomendar_usuario(altura, peso, grasa_corporal)

        for widget in self.recetas_frame.winfo_children():
            widget.destroy()
        for widget in self.ejercicios_frame.winfo_children():
            widget.destroy()

        # Mostrar recetas recomendadas
        for index, row in recetas.iterrows():
            nombre_receta = row['nombre']
            if nombre_receta == 'Batido de Proteína y Plátano':
                imagen_path = 'C:/Users/valen/Downloads/Batido de Proteína y Plátano.jpg'
                self.batido_image = self.load_image(imagen_path)
                label = ttk.Label(self.recetas_frame, image=self.batido_image, text=nombre_receta, compound=tk.TOP)
            elif nombre_receta == 'Wrap de Pollo con Aguacate y Espinacas':
                imagen_path = 'C:/Users/valen/Downloads/Wrap de Pollo con Aguacate y Espinacas.jpg'
                self.wrap_image = self.load_image(imagen_path)
                label = ttk.Label(self.recetas_frame, image=self.wrap_image, text=nombre_receta, compound=tk.TOP)
            elif nombre_receta == 'Filete de Res con Puré de Papas y Ensalada':
                imagen_path = 'C:/Users/valen/Downloads/Filete de Res con Puré de Papas y Ensalada.jpg'
                self.filete_image = self.load_image(imagen_path)
                label = ttk.Label(self.recetas_frame, image=self.filete_image, text=nombre_receta, compound=tk.TOP)
            elif nombre_receta == 'Avena con Frutas':
                imagen_path = 'C:/Users/valen/Downloads/Avena con Frutas.jpg'
                self.avena_image = self.load_image(imagen_path)
                label = ttk.Label(self.recetas_frame, image=self.avena_image, text=nombre_receta, compound=tk.TOP)
            elif nombre_receta == 'Sopa de Pollo y Verduras':
                imagen_path = 'C:/Users/valen/Downloads/Sopa de Pollo y Verduras.jpg'
                self.sopa_image = self.load_image(imagen_path)
                label = ttk.Label(self.recetas_frame, image=self.sopa_image, text=nombre_receta, compound=tk.TOP)
            elif nombre_receta == 'Pechuga de Pollo al Limón con Brócoli':
                imagen_path = 'C:/Users/valen/Downloads/Pechuga de Pollo al Limón con Brócoli.jpg'
                self.pechuga_image = self.load_image(imagen_path)
                label = ttk.Label(self.recetas_frame, image=self.pechuga_image, text=nombre_receta, compound=tk.TOP)
            elif nombre_receta == 'Tostadas de Aguacate con Huevo':
                imagen_path = 'C:/Users/valen/Downloads/Tostadas de Aguacate con Huevo.jpg'
                self.tostadas_image = self.load_image(imagen_path)
                label = ttk.Label(self.recetas_frame, image=self.tostadas_image, text=nombre_receta, compound=tk.TOP)
            elif nombre_receta == 'Salmón al Horno con Batata y Espárragos':
                imagen_path = 'C:/Users/valen/Downloads/Salmón al Horno con Batata y Espárragos.jpg'
                self.salmon_image = self.load_image(imagen_path)
                label = ttk.Label(self.recetas_frame, image=self.salmon_image, text=nombre_receta, compound=tk.TOP)
            elif nombre_receta == 'Hamburguesa de Pavo con Patatas Fritas':
                imagen_path = 'C:/Users/valen/Downloads/Hamburguesa de Pavo con Patatas Fritas.jpg'
                self.hamburguesa_image = self.load_image(imagen_path)
                label = ttk.Label(self.recetas_frame, image=self.hamburguesa_image, text=nombre_receta, compound=tk.TOP)
            else:
                continue

            label.pack(side=tk.LEFT, padx=10, pady=10)
            label.bind("<Button-1>", lambda e, r=row: self.mostrar_receta(r))

        # Mostrar ejercicios recomendados
        for index, row in ejercicios.iterrows():
            nombre_ejercicio = row['nombre']
            if nombre_ejercicio == 'Peso muerto':
                imagen_path = 'C:/Users/valen/Downloads/Peso muerto.jpg'
                self.peso_muerto_image = self.load_image(imagen_path)
                label = ttk.Label(self.ejercicios_frame, image=self.peso_muerto_image, text=nombre_ejercicio,
                                  compound=tk.TOP)
            elif nombre_ejercicio == 'Correr o trotar':
                imagen_path = 'C:/Users/valen/Downloads/Correr o trotar.jpg'
                self.correr_image = self.load_image(imagen_path)
                label = ttk.Label(self.ejercicios_frame, image=self.correr_image, text=nombre_ejercicio,
                                  compound=tk.TOP)
            elif nombre_ejercicio == 'Saltar la cuerda':
                imagen_path = 'C:/Users/valen/Downloads/Saltar la cuerda.jpg'
                self.saltar_cuerda_image = self.load_image(imagen_path)
                label = ttk.Label(self.ejercicios_frame, image=self.saltar_cuerda_image, text=nombre_ejercicio,
                                  compound=tk.TOP)
            elif nombre_ejercicio == 'Sentadillas con peso':
                imagen_path = 'C:/Users/valen/Downloads/Sentadillas con peso.jpg'
                self.sentadillas_image = self.load_image(imagen_path)
                label = ttk.Label(self.ejercicios_frame, image=self.sentadillas_image, text=nombre_ejercicio,
                                  compound=tk.TOP)
            elif nombre_ejercicio == 'Burpees':
                imagen_path = 'C:/Users/valen/Downloads/Burpees.jpg'
                self.burpees_image = self.load_image(imagen_path)
                label = ttk.Label(self.ejercicios_frame, image=self.burpees_image, text=nombre_ejercicio,
                                  compound=tk.TOP)
            elif nombre_ejercicio == 'Flexiones':
                imagen_path = 'C:/Users/valen/Downloads/Flexiones.jpg'
                self.flexiones_image = self.load_image(imagen_path)
                label = ttk.Label(self.ejercicios_frame, image=self.flexiones_image, text=nombre_ejercicio,
                                  compound=tk.TOP)
            elif nombre_ejercicio == 'Dominadas':
                imagen_path = 'C:/Users/valen/Downloads/Dominadas.jpg'
                self.dominadas_image = self.load_image(imagen_path)
                label = ttk.Label(self.ejercicios_frame, image=self.dominadas_image, text=nombre_ejercicio,
                                  compound=tk.TOP)
            elif nombre_ejercicio == 'Press de banca':
                imagen_path = 'C:/Users/valen/Downloads/Press de banca.jpg'
                self.press_banca_image = self.load_image(imagen_path)
                label = ttk.Label(self.ejercicios_frame, image=self.press_banca_image, text=nombre_ejercicio,
                                  compound=tk.TOP)
            elif nombre_ejercicio == 'Fondos en paralelas':
                imagen_path = 'C:/Users/valen/Downloads/Fondos en paralelas.jpg'
                self.fondos_image = self.load_image(imagen_path)
                label = ttk.Label(self.ejercicios_frame, image=self.fondos_image, text=nombre_ejercicio,
                                  compound=tk.TOP)
            elif nombre_ejercicio == 'Ciclismo':
                imagen_path = 'C:/Users/valen/Downloads/Ciclismo.jpg'
                self.ciclismo_image = self.load_image(imagen_path)
                label = ttk.Label(self.ejercicios_frame, image=self.ciclismo_image, text=nombre_ejercicio,
                                  compound=tk.TOP)
            elif nombre_ejercicio == 'Entrenamiento de Intervalos de Alta Intensidad':
                imagen_path = 'C:/Users/valen/Downloads/Entrenamiento de Intervalos de Alta Intensidad.jpg'
                self.hiit_image = self.load_image(imagen_path)
                label = ttk.Label(self.ejercicios_frame, image=self.hiit_image, text=nombre_ejercicio, compound=tk.TOP)
            elif nombre_ejercicio == 'Mountain Climbers':
                imagen_path = 'C:/Users/valen/Downloads/Mountain Climbers.jpg'
                self.mountain_climbers_image = self.load_image(imagen_path)
                label = ttk.Label(self.ejercicios_frame, image=self.mountain_climbers_image, text=nombre_ejercicio,
                                  compound=tk.TOP)
            elif nombre_ejercicio == 'Press de Banca con Mancuernas':
                imagen_path = 'C:/Users/valen/Downloads/Press de Banca con Mancuernas.jpg'
                self.press_mancuernas_image = self.load_image(imagen_path)
                label = ttk.Label(self.ejercicios_frame, image=self.press_mancuernas_image, text=nombre_ejercicio,
                                  compound=tk.TOP)
            elif nombre_ejercicio == 'Peso Muerto Rumano':
                imagen_path = 'C:/Users/valen/Downloads/Peso Muerto Rumano.jpg'
                self.peso_muerto_rumano_image = self.load_image(imagen_path)
                label = ttk.Label(self.ejercicios_frame, image=self.peso_muerto_rumano_image, text=nombre_ejercicio,
                                  compound=tk.TOP)
            elif nombre_ejercicio == 'Flexiones Explosivas':
                imagen_path = 'C:/Users/valen/Downloads/Flexiones Explosivas.jpg'
                self.flexiones_explosivas_image = self.load_image(imagen_path)
                label = ttk.Label(self.ejercicios_frame, image=self.flexiones_explosivas_image, text=nombre_ejercicio,
                                  compound=tk.TOP)
            else:
                continue

            label.pack(side=tk.LEFT, padx=10, pady=10)
            label.bind("<Button-1>", lambda e, r=row: self.mostrar_ejercicio(r))

    def mostrar_receta(self, receta):
        nombre = receta['nombre']
        ingredientes = receta['ingredientes'].replace('. ', '.\n').replace(', ', ',\n')
        preparacion = receta['preparacion'].replace('. ', '.\n').replace(', ', ',\n')
        detalles = f"Ingredientes:\n{ingredientes}\n\nPreparación:\n{preparacion}"
        self.abrir_ventana_detalles(nombre, detalles)

    def mostrar_ejercicio(self, ejercicio):
        nombre = ejercicio['nombre']
        descripcion = ejercicio['descripcion'].replace('. ', '.\n').replace(', ', ',\n')
        duracion = ejercicio['duracion']
        detalles = f"Descripción:\n{descripcion}\n\nDuración:\n{duracion}"
        self.abrir_ventana_detalles(nombre, detalles)

    def abrir_ventana_detalles(self, nombre, detalles):
        ventana_detalles = tk.Toplevel(self.root)
        ventana_detalles.title(nombre)
        ventana_detalles.geometry("400x400")

        detalles_label = ttk.Label(ventana_detalles, text=nombre, font=('Arial', 14))
        detalles_label.pack(pady=10)

        detalles_text = tk.Text(ventana_detalles, height=15, width=50, wrap='word')
        detalles_text.pack(padx=10, pady=10)
        detalles_text.insert(tk.END, detalles)
        detalles_text.config(state=tk.DISABLED)


# Ejecutar la aplicación
root = tk.Tk()
app = App(root)
root.mainloop()