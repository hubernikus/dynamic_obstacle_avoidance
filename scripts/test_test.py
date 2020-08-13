
# x = 'dont'

def inside_loop():
    print(' got variable')

    print(x)

def medium_loop():
    print('make a medium step')
    inside_loop()

    print('step done')

def outside_loop():
    global x
    x = "got ya"
    print('x')

    print('outside is past')
    medium_loop()

class myclass():
    def __init__(self):
        print('init')

    def run(self):
        print('another time I ', x )


outside_loop()
print('I ' + x)


print('and now class')

newinstance = myclass()
newinstance.run()
