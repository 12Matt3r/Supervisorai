from PIL import Image, ImageDraw, ImageFont
import os

def create_icon(size, filename):
    # Create a new image with RGBA (transparent background)
    img = Image.new('RGBA', (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    
    # Background circle with gradient effect
    center = size // 2
    radius = int(size * 0.45)
    
    # Draw background circle
    draw.ellipse([center-radius, center-radius, center+radius, center+radius], 
                fill=(66, 133, 244, 255), outline=(255, 255, 255, 255), width=2)
    
    # Draw robot face
    face_size = int(radius * 0.8)
    face_top = center - face_size // 2
    face_left = center - face_size // 2
    
    # Robot head
    draw.rounded_rectangle([face_left, face_top, face_left+face_size, face_top+face_size], 
                          radius=face_size//6, fill=(255, 255, 255, 255), outline=(51, 51, 51, 255), width=1)
    
    # Eyes
    eye_size = max(2, size // 20)
    eye_y = face_top + face_size // 3
    eye1_x = face_left + face_size // 3
    eye2_x = face_left + 2 * face_size // 3
    
    draw.ellipse([eye1_x-eye_size, eye_y-eye_size, eye1_x+eye_size, eye_y+eye_size], 
                fill=(66, 133, 244, 255))
    draw.ellipse([eye2_x-eye_size, eye_y-eye_size, eye2_x+eye_size, eye_y+eye_size], 
                fill=(66, 133, 244, 255))
    
    # Mouth
    mouth_width = face_size // 3
    mouth_height = max(1, size // 30)
    mouth_x = face_left + face_size // 2 - mouth_width // 2
    mouth_y = face_top + 2 * face_size // 3
    
    draw.rounded_rectangle([mouth_x, mouth_y, mouth_x+mouth_width, mouth_y+mouth_height],
                          radius=mouth_height//2, fill=(51, 51, 51, 255))
    
    # Antenna
    antenna_x = center
    antenna_top = face_top - size // 10
    draw.line([antenna_x, face_top, antenna_x, antenna_top], fill=(51, 51, 51, 255), width=max(1, size//40))
    
    # Antenna tip
    tip_size = max(1, size // 25)
    draw.ellipse([antenna_x-tip_size, antenna_top-tip_size, antenna_x+tip_size, antenna_top+tip_size],
                fill=(255, 107, 53, 255))
    
    # Supervisor badge (S)
    if size >= 32:
        badge_size = size // 6
        badge_x = center + radius - badge_size
        badge_y = center + radius - badge_size
        
        draw.ellipse([badge_x, badge_y, badge_x+badge_size*2, badge_y+badge_size*2],
                    fill=(255, 107, 53, 255), outline=(255, 255, 255, 255), width=1)
        
        # Try to draw 'S' - simplified for small sizes
        if size >= 48:
            try:
                font = ImageFont.load_default()
                draw.text((badge_x+badge_size, badge_y+badge_size), 'S', 
                         fill=(255, 255, 255, 255), anchor='mm', font=font)
            except:
                # If font loading fails, draw a simple shape
                pass
    
    # Save the image
    img.save(filename, 'PNG', optimize=True)

# Generate icons
create_icon(16, 'icon-16.png')
create_icon(48, 'icon-48.png') 
create_icon(128, 'icon-128.png')

print("Icons generated successfully!")
