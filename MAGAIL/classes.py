class Player:
    def __init__(self, id, name, jersey):
        self.id = id
        self.name = name
        self.jersey = jersey
        self.positions = []
    def __str__(self):
        return f"{self.name} ({self.jersey})"
    def addPosition(self, x, y):
        self.positions.append((x, y))