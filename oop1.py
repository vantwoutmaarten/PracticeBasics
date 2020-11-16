class Pet:
    number_of_pets = 0
    def __init__(self, name, age):
        self.name = name
        self.age = age
        Pet.number_of_pets += 1

    def show(self):
        print(f"I am {self.name} and I am {self.age} years old and I am {self.color}")
        Pet.number_of_pets += 1
    
    def speak(self):
        print("I dont know how to speak")
    
class Cat(Pet):
    def __init__(self, name, age,  color):
        super().__init__(name, age)
        self.color = color

    def speak(self):
        print("Meow")
class Dog(Pet):
    def speak(self):
        print("bark")
Pet.number_of_pets = 4

p = Cat("tim", 19, "red")
d = Dog("dogogo", 19)
p.show()


print(p.number_of_pets)
