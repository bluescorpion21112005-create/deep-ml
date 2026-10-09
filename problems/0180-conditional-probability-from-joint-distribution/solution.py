def conditional_probability(joint_distribution: dict) -> float:
    # Kalitlar tuple ekanligini hisobga olamiz: ('A', 'B') va ('`A', 'B')
    p_a_va_b = joint_distribution.get(('A', 'B'), 0.0)
    p_a_yuq_b = joint_distribution.get(('`A', 'B'), 0.0)

    # B ning umumiy ehtimoli
    p_b = p_a_va_b + p_a_yuq_b

    # Nolga bo'lishning oldini olamiz
    if p_b == 0:
        return 0.0

    # Natijani hisoblab, 4 xona aniqlikda yaxlitlaymiz
    result = p_a_va_b / p_b
    return round(result, 4)