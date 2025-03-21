document.addEventListener("DOMContentLoaded", function () {
    const fileInput = document.getElementById("file");
    const previewImage = document.getElementById("preview");

    fileInput.addEventListener("change", function (event) {
        const file = event.target.files[0];
        
        if (file) {
            const reader = new FileReader();
            reader.onload = function (e) {
                previewImage.src = e.target.result;
                previewImage.style.display = "block";
            };
            reader.readAsDataURL(file);
        } else {
            previewImage.style.display = "none";
        }
    });
});
