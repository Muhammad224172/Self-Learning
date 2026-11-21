#make sure the file is named in lowwr case

class Car: 

    def __init__(self , model , year , color , for_sale ):   #contructor #dunder method #self means this object rn

        self.model = model 
        self.year = year
        self.color = color 
        self.for_sale = for_sale

    def drive(self): 
        print(f"You drive the {self.model}")

    def stop(self): 
        print(f"You stopped the {self.model}")

    def describe(self): 
        if self.for_sale: 
            sale = "for sale"
        else: 
            sale = "not for sale"
        print(f"This is a {self.color}  {self.model} produced in {self.year} and it is {sale}")