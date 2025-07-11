# Prueba simple de operaciones de conjuntos
def test_union():
    # Crear conjuntos de prueba
    set1 = set()
    set2 = set()
    
    # Agregar figuras de prueba
    figura1 = ("Círculo", "Rojo")
    figura2 = ("Cuadrado", "Azul")
    figura3 = ("Círculo", "Rojo")  # Duplicado
    
    set1.add(figura1)
    set2.add(figura2)
    set2.add(figura3)
    
    print(f"set1: {set1}")
    print(f"set2: {set2}")
    
    # Probar unión
    union_result = set1.union(set2)
    print(f"Unión: {union_result}")
    print(f"Número de elementos en unión: {len(union_result)}")
    
    # Probar intersección
    intersection_result = set1.intersection(set2)
    print(f"Intersección: {intersection_result}")
    
    # Probar diferencia
    difference_result = set1.difference(set2)
    print(f"Diferencia (set1 - set2): {difference_result}")
    
    # Probar diferencia simétrica
    symmetric_difference_result = set1.symmetric_difference(set2)
    print(f"Diferencia simétrica: {symmetric_difference_result}")

if __name__ == "__main__":
    test_union() 