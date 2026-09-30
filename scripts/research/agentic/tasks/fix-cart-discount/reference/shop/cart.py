"""A minimal shopping cart."""

from .pricing import apply_discount


class Cart:
    def __init__(self):
        self.items = {}  # sku -> [unit_price_cents, quantity]

    def add(self, sku, unit_price_cents, qty=1):
        if qty <= 0:
            raise ValueError("qty must be positive")
        if sku in self.items:
            self.items[sku][1] += qty
        else:
            self.items[sku] = [unit_price_cents, qty]

    def remove(self, sku, qty=1):
        if sku not in self.items:
            raise KeyError(sku)
        self.items[sku][1] -= qty
        if self.items[sku][1] <= 0:
            del self.items[sku]

    def line_count(self):
        return len(self.items)

    def quantity(self, sku):
        return self.items[sku][1] if sku in self.items else 0

    def subtotal_cents(self):
        return sum(p * q for p, q in self.items.values())

    def total_cents(self, code=None):
        return apply_discount(self.subtotal_cents(), code)
