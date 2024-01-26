def test_int(s):
    # check if a string is an actual integer
    try:
        test_value = int(s) 
        # check if it is float
        if str(test_value) != s:
            return False
        else:
            return True
    except ValueError:
        return False

def test_float(s):
    # check if a string is an actual integer
    try:
        test_value = float(s) 
        return True
    except ValueError:
        return False