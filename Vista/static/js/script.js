let sleepCounter = 0; // Contador de detecciones de sueño consecutivas
        const sleepThreshold = 3; // Cuántas veces debe aparecer "Dormido" antes de mostrar la alerta

        function connectToCar() {
            alert("🔗 Conectando al sistema del automóvil...");
            // Aquí puedes agregar la lógica para la conexión con el auto.
        }

        function activateAlarm() {
            alert("🚨 ¡Alarma activada! Despierta.");
            // Aquí puedes agregar la lógica para activar una alerta en el vehículo.
        }

        function showWarning() {
            document.getElementById("warning-screen").style.display = "flex";
        }

        function closeWarning() {
            document.getElementById("warning-screen").style.display = "none";
        }

        function updateStatus() {
            const statusElement = document.getElementById('status');
            const videoFeed = document.getElementById('video-feed');
            const random = Math.random();

            if (random > 0.7) {
                statusElement.textContent = "🟢 Despierto";
                statusElement.classList.remove('sleepy');
                statusElement.classList.add('awake');
                videoFeed.style.transform = "scale(1.0)";
                sleepCounter = 0; // Reiniciar el contador cuando está despierto
                closeWarning();
            } else {
                sleepCounter++; // Aumenta el contador solo si sigue "dormido"
                
                if (sleepCounter >= sleepThreshold) {
                    statusElement.textContent = "🔴 Durmiendo";
                    statusElement.classList.remove('awake');
                    statusElement.classList.add('sleepy');
                    videoFeed.style.transform = "scale(1.1)";  // Zoom para enfocar más la cara
                    showWarning();
                } else {
                    statusElement.textContent = "🟡 Analizando...";
                    statusElement.classList.remove('awake', 'sleepy');
                    videoFeed.style.transform = "scale(1.05)";
                }
            }
        }

    setInterval(updateStatus, 3000);

    document.addEventListener("DOMContentLoaded", function() {
        setInterval(updateStatus, 3000);
    });
    



    