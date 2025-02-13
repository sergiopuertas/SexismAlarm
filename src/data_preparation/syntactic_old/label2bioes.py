import sys


def convert_format(input_text):
    """
    Convierte el formato de etiquetas al formato BMES con manejo completo de la lógica
    y saltos de línea apropiados.
    """
    # Inicializar variables
    result = []
    current_sentence = []

    # Procesar el texto línea por línea
    lines = input_text.strip().split("\n")

    for line in lines:
        line = line.strip()

        # Manejar líneas BOS/EOS
        if "-BOS-" in line:
            current_sentence = []
            continue

        if "-EOS-" in line:
            # Procesar la oración actual
            if current_sentence:
                result.extend(process_sentence(current_sentence))
            result.append("")  # Agregar línea en blanco después de EOS
            current_sentence = []
            continue

        # Procesar líneas normales
        if line and not line.isspace():
            parts = line.split()
            if len(parts) >= 3:
                word = parts[0]
                tag = (
                    parts[2].split("_")[1]
                    if "_" in parts[2] and len(parts[2].split("_")) > 1
                    else "O"
                )  # Obtener la etiqueta después del '_'
                current_sentence.append((word, tag))

    # Procesar última oración si existe
    if current_sentence:
        result.extend(process_sentence(current_sentence))

    return "\n".join(result)


def process_sentence(sentence):
    """
    Procesa una oración completa para determinar las etiquetas BMES.
    """
    result = []

    for i, (word, tag) in enumerate(sentence):
        # Determinar si es una entidad nombrada
        if tag in [
            "nsubj",
            "obj",
            "iobj",
            "root",
            "amod",
            "advmod",
            "csubj",
        ]:  # Ajustar según tus necesidades
            # Si es una entidad de un solo token
            if (i == 0 or sentence[i - 1][1] != tag) and (
                i == len(sentence) - 1 or sentence[i + 1][1] != tag
            ):
                new_tag = f"S-{tag}"

            # Si es el comienzo de una entidad multi-token
            elif i == 0 or sentence[i - 1][1] != tag:
                new_tag = f"B-{tag}"

            # Si es el final de una entidad multi-token
            elif i == len(sentence) - 1 or sentence[i + 1][1] != tag:
                new_tag = f"E-{tag}"

            # Si es parte intermedia de una entidad multi-token
            else:
                new_tag = f"M-{tag}"

        # Para etiquetas sintácticas (como nsubj, amod, etc.), usar 'O'
        else:
            new_tag = "O"

        # Agregar la palabra y su etiqueta al resultado
        result.append(f"{word} {new_tag}")

    return result


def main():
    # Leer el archivo de entrada
    try:
        with open("selected.labels", "r", encoding="utf-8") as f:
            input_text = f.read()
    except FileNotFoundError:
        print("Error: No se encontró el archivo 'selected.labels'")
        sys.exit(1)
    except Exception as e:
        print(f"Error al leer el archivo: {str(e)}")
        sys.exit(1)

    # Convertir el formato
    converted_text = convert_format(input_text)

    # Escribir el resultado en un nuevo archivo
    try:
        with open("output.bmes", "w", encoding="utf-8") as f:
            f.write(converted_text)
        print("Conversión completada. El resultado se ha guardado en 'output.bmes'")
    except Exception as e:
        print(f"Error al escribir el archivo de salida: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
