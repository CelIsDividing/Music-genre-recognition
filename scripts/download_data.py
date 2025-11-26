import os
import yt_dlp

# Primer YouTube linkova za svaki žanr (dodajte svoje linkove!)
genre_links = {
    "blues": [
        "https://youtu.be/ioOzsi9aHQQ?list=PLFjmckBbDlzSE47xD0FivigDOARiLJl61",
    ],
    "classical": [
        "https://youtu.be/9E6b3swbnWg?list=PLO_1AmtK1TMRi01-V_tdHKDLBu7cNWIgf"
    ],
    "electronic": [
        "https://youtu.be/foE1mO2yM04?list=PL7wr9BYcCCyNb0IhqqebdLEMkklMSOttA"
    ],
    "folk": [
        "https://youtu.be/5NnINMjbJCU?list=PL3QM09EUkf8Rht2yMxILxj53iJS2gwrEX"
    ],
    "metal": [
        "https://youtu.be/CD-E-LDc384?list=PLGBuKfnErZlBQRS7AqbpA8N-_qO50ReTr"
    ],
    "pop": [
        "https://youtu.be/WcIcVapfqXw?list=PLeME5BS6gE76Vmr1PubWrLa1cbCr7uOle"
    ],
    "rock": [
        "https://youtu.be/kXYiU_JCYtU?list=PLnAYe9y3M4Cbf-PZCjEs9vJU-OAMqk4o6"
    ],
    "jazz": [
        "https://youtu.be/ZZcuSBouhVA?list=RDATgyfhdHlwZV9taXg"
    ],
    "rap": [
        "https://youtu.be/begbi44ZLqw?list=PLDIoUOhQQPlXFSnCfj8HuVhOUSC0QwxYD"
    ],
    "country": [
        "https://youtu.be/uKiF-UMnB9A?list=PLD4LoGCGqHHj-73CZkstinij6xSnzoMaR"
    ]
}

for genre, links in genre_links.items():
    genre_dir = f"data/raw/{genre}"
    os.makedirs(genre_dir, exist_ok=True)
    for url in links:
        ydl_opts = {
            'format': 'bestaudio/best',
            'outtmpl': f'{genre_dir}/%(title)s.%(ext)s',
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'wav',
                'preferredquality': '192',
            }],
            'playlist_items': '1-40',  # Preuzmi samo prvih 40 pesama iz plejliste
            'ignoreerrors': True,  # Ignoriši greške i nastavi sa sledećim videoima
            'quiet': False
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])