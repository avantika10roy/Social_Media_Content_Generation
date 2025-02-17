def get_color_themes():
    color_themes = {
        # Classic Themes
        "black-gold": [(0, 0, 0), (255, 215, 0)],  # Black and Gold
        "blue-silver": [(0, 0, 255), (192, 192, 192)],  # Blue and Silver
        "red-white": [(255, 0, 0), (255, 255, 255)],  # Red and White
        "green-brown": [(0, 128, 0), (139, 69, 19)],  # Green and Brown
        "purple-pink": [(128, 0, 128), (255, 192, 203)],  # Purple and Pink
        "teal-orange": [(0, 128, 128), (255, 165, 0)],  # Teal and Orange
        "navy-yellow": [(0, 0, 128), (255, 255, 0)],  # Navy and Yellow
        "gray-lime": [(128, 128, 128), (0, 255, 0)],  # Gray and Lime

        # Pastel Themes
        "pastel-blue-pink": [(173, 216, 230), (255, 182, 193)],  # Pastel Blue and Pink
        "pastel-green-lavender": [(152, 251, 152), (230, 230, 250)],  # Pastel Green and Lavender
        "pastel-peach-mint": [(255, 218, 185), (189, 252, 201)],  # Pastel Peach and Mint
        "pastel-yellow-lime": [(255, 255, 224), (204, 255, 204)],  # Pastel Yellow and Lime
        "pastel-purple-aqua": [(216, 191, 216), (173, 216, 230)],  # Pastel Purple and Aqua

        # Earthy Themes
        "earthy-tan-green": [(210, 180, 140), (34, 139, 34)],  # Earthy Tan and Green
        "earthy-burgundy-cream": [(128, 0, 32), (255, 253, 208)],  # Earthy Burgundy and Cream
        "earthy-olive-sand": [(128, 128, 0), (244, 164, 96)],  # Earthy Olive and Sand
        "earthy-terracotta-brown": [(204, 102, 51), (139, 69, 19)],  # Earthy Terracotta and Brown
        "earthy-moss-rust": [(98, 123, 52), (183, 65, 14)],  # Earthy Moss and Rust

        # Bright Themes
        "bright-pink-yellow": [(255, 105, 180), (255, 255, 0)],  # Bright Pink and Yellow
        "bright-orange-blue": [(255, 165, 0), (0, 0, 255)],  # Bright Orange and Blue
        "bright-red-cyan": [(255, 0, 0), (0, 255, 255)],  # Bright Red and Cyan
        "bright-purple-green": [(128, 0, 128), (0, 255, 0)],  # Bright Purple and Green
        "bright-teal-magenta": [(0, 128, 128), (255, 0, 255)],  # Bright Teal and Magenta

        # Neutral Themes
        "neutral-beige-gray": [(245, 245, 220), (169, 169, 169)],  # Neutral Beige and Gray
        "neutral-chocolate-cream": [(123, 63, 0), (255, 253, 208)],  # Neutral Chocolate and Cream
        "neutral-black-white": [(0, 0, 0), (255, 255, 255)],  # Neutral Black and White
        "neutral-silver-charcoal": [(192, 192, 192), (54, 69, 79)],  # Neutral Silver and Charcoal
        "neutral-mocha-cream": [(186, 148, 123), (255, 253, 208)],  # Neutral Mocha and Cream

        # Vintage Themes
        "vintage-burgundy-gold": [(128, 0, 32), (255, 215, 0)],  # Vintage Burgundy and Gold
        "vintage-turquoise-beige": [(64, 224, 208), (245, 245, 220)],  # Vintage Turquoise and Beige
        "vintage-navy-red": [(0, 0, 128), (255, 0, 0)],  # Vintage Navy and Red
        "vintage-mustard-darkgreen": [(255, 219, 88), (0, 100, 0)],  # Vintage Mustard and Dark Green
        "vintage-rosewood-teal": [(101, 67, 33), (0, 128, 128)],  # Vintage Rosewood and Teal

        # Summer Themes
        "summer-sky-sun": [(135, 206, 235), (255, 223, 0)],  # Summer Sky and Sun
        "summer-sunset-orange-pink": [(255, 94, 77), (255, 182, 193)],  # Summer Sunset Orange and Pink
        "summer-seafoam-coral": [(46, 139, 87), (255, 127, 80)],  # Summer Seafoam and Coral
        "summer-sand-lagoon": [(255, 229, 204), (0, 128, 128)],  # Summer Sand and Lagoon
        "summer-pineapple-turquoise": [(255, 221, 51), (64, 224, 208)],  # Summer Pineapple and Turquoise

        # Winter Themes
        "winter-iceblue-silver": [(173, 216, 230), (192, 192, 192)],  # Winter Ice Blue and Silver
        "winter-snow-red": [(255, 250, 250), (255, 0, 0)],  # Winter Snow and Red
        "winter-darkgreen-ivory": [(0, 100, 0), (255, 255, 240)],  # Winter Dark Green and Ivory
        "winter-midnight-blue-white": [(25, 25, 112), (255, 255, 255)],  # Winter Midnight Blue and White
        "winter-gray-icyblue": [(169, 169, 169), (176, 224, 230)],  # Winter Gray and Icy Blue

        # Retro Themes
        "retro-orange-brown": [(255, 69, 0), (139, 69, 19)],  # Retro Orange and Brown
        "retro-aqua-pink": [(0, 255, 255), (255, 105, 180)],  # Retro Aqua and Pink
        "retro-yellow-red": [(255, 255, 0), (255, 0, 0)],  # Retro Yellow and Red
        "retro-purple-yellow": [(128, 0, 128), (255, 255, 0)],  # Retro Purple and Yellow
        "retro-teal-coral": [(0, 128, 128), (255, 127, 80)],  # Retro Teal and Coral

        # Bold Themes
        "bold-black-cyan": [(0, 0, 0), (0, 255, 255)],  # Bold Black and Cyan
        "bold-orange-pink": [(255, 165, 0), (255, 105, 180)],  # Bold Orange and Pink
        "bold-red-turquoise": [(255, 0, 0), (64, 224, 208)],  # Bold Red and Turquoise
        "bold-purple-yellow": [(128, 0, 128), (255, 255, 0)],  # Bold Purple and Yellow
        "bold-teal-green": [(0, 128, 128), (0, 255, 0)],  # Bold Teal and Green

        # Minimalist Themes
        "minimalist-white-black": [(255, 255, 255), (0, 0, 0)],  # Minimalist White and Black
        "minimalist-gray-white": [(169, 169, 169), (255, 255, 255)],  # Minimalist Gray and White
        "minimalist-navy-cream": [(0, 0, 128), (255, 253, 208)],  # Minimalist Navy and Cream
        "minimalist-beige-gray": [(245, 245, 220), (169, 169, 169)],  # Minimalist Beige and Gray
        "minimalist-charcoal-white": [(54, 69, 79), (255, 255, 255)],  # Minimalist Charcoal and White

        # Elegant Themes
        "elegant-black-diamond": [(0, 0, 0), (255, 185, 15)],  # Elegant Black and Diamond
        "elegant-emerald-gold": [(80, 200, 120), (255, 215, 0)],  # Elegant Emerald and Gold
        "elegant-royal-blue-silver": [(65, 105, 225), (192, 192, 192)],  # Elegant Royal Blue and Silver
        "elegant-platinum-pink": [(229, 228, 226), (255, 182, 193)],  # Elegant Platinum and Pink
        "elegant-dark-purple-cream": [(128, 0, 128), (255, 253, 208)],  # Elegant Dark Purple and Cream

        # Rustic Themes
        "rustic-brick-sky": [(178, 34, 34), (135, 206, 235)],  # Rustic Brick and Sky
        "rustic-forest-honey": [(34, 139, 34), (255, 223, 0)],  # Rustic Forest and Honey
        "rustic-copper-moss": [(184, 115, 51), (98, 123, 52)],  # Rustic Copper and Moss
        "rustic-oak-cream": [(139, 69, 19), (255, 253, 208)],  # Rustic Oak and Cream
        "rustic-wine-clay": [(128, 0, 32), (204, 102, 51)],  # Rustic Wine and Clay

        # Soft Themes
        "soft-lavender-paleyellow": [(230, 230, 250), (255, 255, 224)],  # Soft Lavender and Pale Yellow
        "soft-pink-mint": [(255, 182, 193), (189, 252, 201)],  # Soft Pink and Mint
        "soft-peach-lilac": [(255, 218, 185), (200, 162, 200)],  # Soft Peach and Lilac
        "soft-sky-lime": [(135, 206, 235), (0, 255, 0)],  # Soft Sky and Lime
        "soft-aqua-powderblue": [(0, 255, 255), (176, 224, 230)],  # Soft Aqua and Powder Blue

        # Tropical Themes
        "tropical-turquoise-pink": [(64, 224, 208), (255, 105, 180)],  # Tropical Turquoise and Pink
        "tropical-orange-lime": [(255, 165, 0), (0, 255, 0)],  # Tropical Orange and Lime
        "tropical-yellow-coral": [(255, 255, 0), (255, 127, 80)],  # Tropical Yellow and Coral
        "tropical-azure-sunset": [(0, 127, 255), (255, 94, 77)],  # Tropical Azure and Sunset
        "tropical-sunshine-palm": [(255, 223, 0), (0, 128, 0)],  # Tropical Sunshine and Palm

        # High Contrast Themes
        "highcontrast-black-white": [(0, 0, 0), (255, 255, 255)],  # High Contrast Black and White
        "highcontrast-red-cyan": [(255, 0, 0), (0, 255, 255)],  # High Contrast Red and Cyan
        "highcontrast-blue-orange": [(0, 0, 255), (255, 165, 0)],  # High Contrast Blue and Orange
        "highcontrast-purple-yellow": [(128, 0, 128), (255, 255, 0)],  # High Contrast Purple and Yellow
        "highcontrast-green-magenta": [(0, 255, 0), (255, 0, 255)],  # High Contrast Green and Magenta

        # Soft Gradient Themes
        "gradient-peach-sunset": [(255, 182, 193), (255, 94, 77)],  # Gradient Peach and Sunset
        "gradient-blue-lavender": [(135, 206, 235), (230, 230, 250)],  # Gradient Blue and Lavender
        "gradient-pink-yellow": [(255, 182, 193), (255, 255, 0)],  # Gradient Pink and Yellow
        "gradient-emerald-ivory": [(80, 200, 120), (255, 255, 240)],  # Gradient Emerald and Ivory
        "gradient-sky-mint": [(135, 206, 235), (189, 252, 201)],  # Gradient Sky and Mint

        # Cosmic Themes
        "cosmic-darkblue-neonpink": [(25, 25, 112), (255, 20, 147)],  # Cosmic Dark Blue and Neon Pink
        "cosmic-purple-neonlime": [(128, 0, 128), (0, 255, 0)],  # Cosmic Purple and Neon Lime
        "cosmic-black-neonblue": [(0, 0, 0), (0, 0, 255)],  # Cosmic Black and Neon Blue
        "cosmic-darkpurple-neonorange": [(75, 0, 130), (255, 69, 0)],  # Cosmic Dark Purple and Neon Orange
        "cosmic-neoncyan-neonyellow": [(0, 255, 255), (255, 255, 0)],  # Cosmic Neon Cyan and Neon Yellow

        # Candy Themes
        "candy-pink-yellow": [(255, 105, 180), (255, 255, 0)],  # Candy Pink and Yellow
        "candy-mint-chocolate": [(189, 252, 201), (123, 63, 0)],  # Candy Mint and Chocolate
        "candy-cottonblue-pastelpink": [(176, 224, 230), (255, 182, 193)],  # Candy Cotton Blue and Pastel Pink
        "candy-lavender-cherry": [(230, 230, 250), (222, 49, 99)],  # Candy Lavender and Cherry
        "candy-orange-lime": [(255, 165, 0), (0, 255, 0)],  # Candy Orange and Lime

        # Futuristic Themes
        "futuristic-silver-black": [(192, 192, 192), (0, 0, 0)],  # Futuristic Silver and Black
        "futuristic-neonblue-silver": [(0, 0, 255), (192, 192, 192)],  # Futuristic Neon Blue and Silver
        "futuristic-robotgrey-neonred": [(169, 169, 169), (255, 0, 0)],  # Futuristic Robot Grey and Neon Red
        "futuristic-metallicpurple-neonlime": [(102, 51, 153), (0, 255, 0)],  # Futuristic Metallic Purple and Neon Lime
        "futuristic-blue-glow": [(0, 0, 255), (0, 255, 255)],  # Futuristic Blue and Glow

        # Natural Themes
        "natural-sand-wood": [(244, 164, 96), (139, 69, 19)],  # Natural Sand and Wood
        "natural-forest-sky": [(34, 139, 34), (135, 206, 235)],  # Natural Forest and Sky
        "natural-sunflower-grass": [(255, 255, 0), (124, 252, 0)],  # Natural Sunflower and Grass
        "natural-rock-moss": [(169, 169, 169), (98, 123, 52)],  # Natural Rock and Moss
        "natural-ocean-sunset": [(0, 0, 255), (255, 94, 77)],  # Natural Ocean and Sunset

        # Retro-Futuristic Themes
        "retro-futuristic-pink-cyan": [(255, 105, 180), (0, 255, 255)],  # Retro-Futuristic Pink and Cyan
        "retro-futuristic-purple-yellow": [(128, 0, 128), (255, 255, 0)],  # Retro-Futuristic Purple and Yellow
        "retro-futuristic-teal-magenta": [(0, 128, 128), (255, 0, 255)],  # Retro-Futuristic Teal and Magenta
        "retro-futuristic-orange-green": [(255, 165, 0), (0, 255, 0)],  # Retro-Futuristic Orange and Green
        "retro-futuristic-blue-red": [(0, 0, 255), (255, 0, 0)],  # Retro-Futuristic Blue and Red

        # Gothic Themes
        "gothic-darkred-black": [(139, 0, 0), (0, 0, 0)],  # Gothic Dark Red and Black
        "gothic-deeppurple-black": [(75, 0, 130), (0, 0, 0)],  # Gothic Deep Purple and Black
        "gothic-vampire-black": [(102, 0, 0), (0, 0, 0)],  # Gothic Vampire Red and Black
        "gothic-lilac-darkgray": [(200, 162, 200), (169, 169, 169)],  # Gothic Lilac and Dark Gray
        "gothic-bloodred-gray": [(139, 0, 0), (169, 169, 169)],  # Gothic Blood Red and Gray

        # Monochrome Themes
        "monochrome-dark": [(0, 0, 0), (64, 64, 64)],  # Monochrome Dark
        "monochrome-light": [(255, 255, 255), (192, 192, 192)],  # Monochrome Light
        "monochrome-gray": [(169, 169, 169), (105, 105, 105)],  # Monochrome Gray
        "monochrome-charcoal": [(54, 69, 79), (128, 128, 128)],  # Monochrome Charcoal
        "monochrome-silver": [(192, 192, 192), (211, 211, 211)],  # Monochrome Silver
        
    }

    return color_themes