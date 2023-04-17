import numpy as np


def extract_intersections(rows, cols, base_row_id) -> list:
    """
    Удаляет строки, пересекающиеся с заданной.
    """

    # буфер извлечённых столбцов
    extracted_cols = []
    for col_id in rows[base_row_id]:
        # вынимаем текущий столбец из таблицы в буфер
        extracted_col = cols.pop(col_id)
        extracted_cols.append(extracted_col)
        # удаляем все пересекающиеся строки из всех оставшихся столбцов
        for intersecting_row_id in extracted_col:
            for other_col_id in rows[intersecting_row_id]:
                if other_col_id != col_id:
                    cols[other_col_id].remove(intersecting_row_id)
    return extracted_cols


def restore_intersections(rows, cols, base_row_id, extracted_cols):
    """
    Возвращает удалённые с помощью extract_intersections строки на место.
    """

    # т.к. удаляли столбцы от первого пересечения с base_row к последнему,
    # то и восстанавливать надо в обратном порядке
    for col_id, extracted_col in zip(reversed(rows[base_row_id]), reversed(extracted_cols)):
        cols[col_id] = extracted_col
        for added_row_id in extracted_col:
            for col in rows[added_row_id]:
                cols[col].add(added_row_id)


def algorithm_x(rows, cols, cover):
    if not cols:
        return cover

    # ищем столбец с минимальным числом элементов
    _, min_col = min(cols.items(), key=lambda x: len(x[1]))
    for row_id in min_col:
        cover.append(row_id)
        # удаляем пересекающиеся подмножества и содержащиеся в строке row_id элементы
        extracted_cols = extract_intersections(rows, cols, row_id)
        s = algorithm_x(rows, cols, cover)
        # если нашлось непустое решение - готово, выходим
        if s != None:
            return s
        # иначе восстанавливаем пересекающиеся подмножества
        restore_intersections(rows, cols, row_id, extracted_cols)
        # удаляем "неудачное" подмножество из покрытия
        del cover[-1]
    # сюда дойдём либо если в min_col пусто, либо когда рекурсивный поиск не нашёл решения
    return None


def solve_sudoku(puzzle: list) -> list:
    """
    Судоку задаётся матрицей 9х9, на месте неизвестных чисел нули.
    """
    assert(len(puzzle) == 81)
    puzzle = np.array(puzzle).reshape(9, 9)

    # проверяем входные данные на корректность
    assert(len(puzzle) == 9)
    for row in range(9):
        assert(len(puzzle[row]) == 9)
        for col in range(9):
            assert (0 <= puzzle[row, col] <= 9)

    # идентификаторы строк - кортежи вида (row, col, num)
    # идентификаторы столбцов:
    # ('a', row, col) - на пересечении row и col стоит число
    # ('b', row, num) - в строке row есть число num
    # ('c', col, num) - в столбце col есть число num
    # ('d', q, num) - в квадранте q есть число num
    rows = dict() # size = 9*9*9
    cols = dict() # size = 4*9*9

    # заполняем строки
    for row in range(1, 10):
        for col in range(1, 10):
            for num in range(1, 10):
                row_id = (row, col, num)
                quad = ((row-1)//3)*3 + (col-1)//3 + 1
                rules = [('a', row, col), ('b', row, num), ('c', col, num), ('d', quad, num)]
                rows[row_id] = rules

    # заполняем столбцы
    for rule in ['a', 'b', 'c', 'd']:
        for n1 in range(1, 10):
            for n2 in range(1, 10):
                cols[(rule, n1, n2)] = set()
    for row_id, row_values in rows.items():
        for value in row_values:
            cols[value].add(row_id)

    # s - заготовка для ответа-покрытия
    # для начала туда надо внести те цифры, которые уже заполнены
    cover = []

    for row in range(1, 10):
        for col in range(1, 10):
            if puzzle[row-1, col-1] != 0:
                true_row_id = (row, col, puzzle[row-1, col-1])
                cover.append(true_row_id)
                # добавив клетку в решение, удаляем из матрицы все несовместимые элементы
                extracted_cols_debug = extract_intersections(rows, cols, true_row_id)

    # всё, что осталось - найти покрытие
    cover = algorithm_x(rows, cols, cover)
    if cover is None:
        raise "Puzzle has no solution!"

    # проверяем, что элементы не дублируются
    assert(len(cover) == 81)

    # переносим ответ в матрицу
    for (row, col, num) in cover:
        # проверяем, что не перезатёрли исходные позиции
        assert(puzzle[row-1, col-1] == 0 or puzzle[row-1, col-1] == num)
        puzzle[row-1, col-1] = num
    # проверяем корректность решения
    for i in range(9):
        assert(sorted(puzzle[i, :]) == [1, 2, 3, 4, 5, 6, 7, 8, 9])
        assert(sorted(puzzle[:, i]) == [1, 2, 3, 4, 5, 6, 7, 8, 9])
        quad_i = (i // 3)*3
        quad_j = (i % 3)*3
        quad = puzzle[quad_i:quad_i+3, quad_j:quad_j+3].flatten().tolist()
        assert(sorted(quad) == [1, 2, 3, 4, 5, 6, 7, 8, 9])

    return puzzle.tolist()

