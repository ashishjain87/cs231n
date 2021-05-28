from Affixer import Affixer
from FixedAffixer import FixedAffixer

def main():
    affixer: Affixer = FixedAffixer()
    (location, scale) = affixer.decide_where_and_scale(None, None, None)
    centerX, centerY = location
    print("centerX", centerX, "centerY", centerY, "scale", scale)

if __name__ == '__main__':
    main()
