document.getElementById('kayit-link').addEventListener('click', function(event) {
    event.preventDefault();
    document.getElementById('giris-form').style.opacity = 0;
    setTimeout(function() {
        document.getElementById('giris-form').style.display = 'none';
        document.getElementById('kayit-form').style.display = 'block';
        setTimeout(function() {
            document.getElementById('kayit-form').style.opacity = 1;
        }, 10);
    }, 500);
});

document.getElementById('giris-link').addEventListener('click', function(event) {
    event.preventDefault();
    document.getElementById('kayit-form').style.opacity = 0;
    setTimeout(function() {
        document.getElementById('kayit-form').style.display = 'none';
        document.getElementById('giris-form').style.display = 'block';
        setTimeout(function() {
            document.getElementById('giris-form').style.opacity = 1;
        }, 10);
    }, 500);
});
