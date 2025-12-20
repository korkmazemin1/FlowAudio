import sounddevice as sd

print("="*40)
print("🔊 SES CİHAZLARI LİSTESİ")
print("="*40)

# List all devices
# The output shows ID, Name, Core Audio API, Input Channels, Output Channels
print(sd.query_devices())

print("\n" + "="*40)
print("🎤 VARSAYILAN CİHAZLAR")
print("="*40)
try:
    default_input = sd.query_devices(kind='input')
    print(f"Varsayılan Giriş (Mic): {default_input['name']} (ID: {default_input['index']})")
except Exception as e:
    print("Varsayılan giriş cihazı bulunamadı!")

print("="*40)