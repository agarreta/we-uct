import os

from we.arguments import Arguments
args = Arguments()

def SMT_eval_Z3(eq, timeout, solver=''):
    w = eq.get_string_form()
    os.chdir('..')
    os.chdir('..')
    os.chdir('..')
    with open('z3-master/build/python/word_equation.txt', 'w+') as f:
        f.write(f'{w}')
    with open('z3-master/build/python/timeout.txt', 'w+') as f:
        f.write(f'{timeout}')
    os.chdir('PycharmProjects/pr/test')

    folder = 'cd\ncd z3-master/build/python'
    if timeout is None:
        command = f'{folder}\npython execute_word_equation{solver}.py'
    else:
        command = f'{folder}\npython execute_word_equation_08_second{solver}.py'
    if False:
        if False: #timeout >= 20000:
            command = f'{folder}\npython execute_word_equation{solver}.py'
        elif timeout == 1200:
            command = f'{folder}\npython execute_word_equation_1_2_second{solver}.py'
        elif timeout == 800:
            command = f'{folder}\npython execute_word_equation_08_second{solver}.py'
        elif timeout == 400:
            command = f'{folder}\npython execute_word_equation_04_second.py'

    stream = os.popen(command)
    result = stream.read()
    #result = result.split('-')
    out = result.split('-')[0]
    print(out)
    if out == 'sat':
        out = 1
    elif out == 'unsat':
        out = -1
    else:
        out = 0
    return out

def SMT_eval_woorpje(eq, timeout):
    if timeout is not None:
        timeout = int(timeout/1000)
    w = eq.get_string_form()
    os.chdir('..')
    os.chdir('..')
    os.chdir('..')
    with open('woorpje-0_2/bin/word_equation.txt', 'w+') as f:
        f.write(f'{w}')
    with open(f'woorpje-0_2/bin/timeout.txt', 'w+') as f:
        f.write(f'{timeout}')
    os.chdir('PycharmProjects/pr/test')

    if timeout is None:
        command = f'cd\ncd woorpje-0_2/bin\npython execute_word_equation.py'
    else:
        command = f'cd\ncd woorpje-0_2/bin\npython execute_word_equation_08_second.py'

    if False:
        if timeout >= 20000:
            command = f'cd\ncd woorpje-0_2/bin\npython execute_word_equation.py'
        elif timeout == 800:
            command = f'cd\ncd woorpje-0_2/bin\npython execute_word_equation_08_second.py'
        elif timeout == 1200:
            command = f'cd\ncd woorpje-0_2/bin\npython execute_word_equation_1_2_second.py'

    stream = os.popen(command)
    result = stream.read()
    print('A')
    print(result)
    print('B')
    out = result
    if 'Found a solution' in out:
        out = 1
    else:
        out = 0
    print(out)
    return out

def SMT_eval_CVC4(eq, timeout):
    w = eq.get_string_form()
    os.chdir('..')
    os.chdir('..')
    os.chdir('..')
    with open('CVC4-master/build/bin/word_equation.txt', 'w+') as f:
        f.write(f'{w}')
    with open(f'CVC4-master/build/bin/timeout.txt', 'w+') as f:
        f.write(f'{timeout}')
    os.chdir('PycharmProjects/pr/test')

    folder = 'cd\ncd CVC4-master/build/bin'
    if timeout is None:
        command = f'{folder}\npython execute_word_equation.py'
    else:
        command = f'{folder}\npython execute_word_equation_08_second.py'
    if False:
        if timeout >= 20000:
            command = f'{folder}\npython execute_word_equation.py'
        elif timeout == 1200:
            command = f'{folder}\npython execute_word_equation_1_2_second.py'
        elif timeout == 800:
            command = f'{folder}\npython execute_word_equation_08_second.py'
        elif timeout == 400:
            command = f'{folder}\npython execute_word_equation_04_second.py'

    stream = os.popen(command)
    out = stream.read()
    print(out)
    print(out.split('\n'))
    out = out.split('\n')[0]
    print(out)
    if out == 'sat':
        out = 1
    elif out == 'unsat':
        out = -1
    else:
        out = 0
    return out

def SMT_eval_sloth(eq, timeout):
    w = eq.w
    os.chdir('..')
    os.chdir('..')
    os.chdir('..')

    folder = 'cd\ncd sloth-1.0'
    if timeout is None:
        command = f'{folder}\npython execute_word_equation.py'
    else:
        command = f'{folder}\npython execute_word_equation_08_second.py'
    if False:
        if timeout >= 20000:
            command = f'{folder}\npython execute_word_equation.py'
        elif timeout == 1200:
            command = f'{folder}\npython execute_word_equation_1_2_second.py'
        elif timeout == 800:
            command = f'{folder}\npython execute_word_equation_08_second.py'
        elif timeout == 400:
            command = f'{folder}\npython execute_word_equation_04_second.py'

    stream = os.popen(command)
    out = stream.read()
    print(out)
    out = out.split('\n')[-3]
    print(out)

    if 'sat' in out and 'unsat' not in out:
        out = 1
    elif 'unknown' in out or 'timeout' in out:
        out = 0
    else:
        out = -1
    return out

def SMT_eval_TRAU(eq, timeout):
    w = eq.get_string_form()
    os.chdir('..')
    os.chdir('..')
    os.chdir('..')
    folder = 'z3-new_trau/trau-build'
    print(folder)
    with open(f'{folder}/word_equation.txt', 'w+') as f:
        f.write(f'{w}')
    with open(f'{folder}/timeout.txt', 'w+') as f:
        f.write(f'{timeout}')
    os.chdir('PycharmProjects/pr/test')

    folder = 'cd\ncd z3-new_trau/trau-build'
    if timeout is None:
        command = f'{folder}\npython execute_word_equation.py'
    else:
        command = f'{folder}\npython execute_word_equation_08_second.py'
    if False:
        if timeout >= 20000:
            command = f'{folder}\npython execute_word_equation.py'
        elif timeout == 1200:
            command = f'{folder}\npython execute_word_equation_1_2_second.py'
        elif timeout == 800:
            command = f'{folder}\npython execute_word_equation_08_second.py'
        elif timeout == 400:
            command = f'{folder}\npython execute_word_equation_04_second.py'

    stream = os.popen(command)
    out = stream.read().split('\n')[0]
    print(out)
    if out == 'sat':
        out = 1
    elif out == 'unsat':
        out = -1
    else:
        out = 0
    return out


def SMT_eval_seq(eq, timeout):
    return SMT_eval_Z3(eq, timeout, solver='_Seq')


def SMT_eval(args, eq, timeout=None):



    if len(eq.w.split('='))==1:
        print('ERROR')

    if args.smt_solver == 'Z3':
        return SMT_eval_Z3(eq, timeout)
    elif args.smt_solver == 'woorpje':
        return SMT_eval_woorpje(eq, timeout)
    elif args.smt_solver == 'CVC4':
        return SMT_eval_CVC4(eq, timeout)
    elif args.smt_solver == 'sloth':
        return SMT_eval_sloth(eq, timeout)
    elif args.smt_solver == 'TRAU':
        return SMT_eval_TRAU(eq, timeout)
    elif args.smt_solver == 'seq':
        return SMT_eval_seq(eq, timeout)
    elif args.smt_solver == 'dumb':
        return eq.sat