window.addEventListener('load', () => {
    const canvas = document.getElementById('drawCanvas');
    const ctx = canvas.getContext('2d');
    const downloadBtn = document.getElementById('downloadBtn');
    const clearBtn = document.getElementById('clearBtn');

    // Canvasのサイズをウィンドウ全体（コントロールバーを除く）に合わせる
    const controlsHeight = document.getElementById('controls').offsetHeight;
    canvas.width = window.innerWidth;
    canvas.height = window.innerHeight - controlsHeight;

    ctx.lineWidth = 2; // 線の太さ
    ctx.strokeStyle = '#000000'; // 線の色
    ctx.lineCap = 'round'; // 線の端を丸く
    ctx.lineJoin = 'round'; // 線の角を丸く

    let isDrawing = false;
    let strokeCount = 0; // ストローク番号
    let drawingData = []; // 全ての座標データを格納する配列

    // ----------------------------------------
    // イベントリスナーの登録
    // ----------------------------------------

    // ペンが触れた時 (書き始め)
    canvas.addEventListener('pointerdown', startDrawing);
    // ペンが動いた時 (書いている最中)
    canvas.addEventListener('pointermove', draw);
    // ペンが離れた時 (書き終わり)
    canvas.addEventListener('pointerup', stopDrawing);
    // ペンがCanvas領域から外れた時
    canvas.addEventListener('pointerleave', stopDrawing);

    // ボタンの処理
    downloadBtn.addEventListener('click', downloadCSV);
    clearBtn.addEventListener('click', clearCanvas);

    // ----------------------------------------
    // 描画とデータ記録の関数
    // ----------------------------------------

    function startDrawing(e) {
        // Apple Pencilからの入力のみを許可 (touchやmouseを除外)
        if (e.pointerType !== 'pen') return;

        isDrawing = true;
        strokeCount++; // 新しいストローク
        
        // 描画の開始位置をセット
        const pos = getCanvasPosition(e);
        ctx.beginPath();
        ctx.moveTo(pos.x, pos.y);

        // データを記録
        recordData(e, 'down');
    }

    function draw(e) {
        if (!isDrawing || e.pointerType !== 'pen') return;

        const pos = getCanvasPosition(e);

        // Canvasに線を描画
        ctx.lineTo(pos.x, pos.y);
        ctx.stroke();

        // データを記録
        recordData(e, 'move');
    }

    function stopDrawing(e) {
        if (e.pointerType !== 'pen') return;
        isDrawing = false;
    }

    // Canvas上の相対座標を取得
    function getCanvasPosition(e) {
        const rect = canvas.getBoundingClientRect();
        return {
            x: e.clientX - rect.left,
            y: e.clientY - rect.top
        };
    }

    // データを配列に格納
    function recordData(e, eventType) {
        const pos = getCanvasPosition(e);
        
        drawingData.push({
            timestamp: e.timeStamp, // ページ読み込みからの経過ミリ秒
            event_type: eventType,  // 'down' または 'move'
            stroke_id: strokeCount, // 何番目のストロークか
            x: pos.x,
            y: pos.y,
            pressure: e.pressure, // 筆圧 (0.0 - 1.0)
            tilt_x: e.tiltX,     // X軸の傾き
            tilt_y: e.tiltY      // Y軸の傾き
        });
    }

    // ----------------------------------------
    // CSV処理とクリアの関数
    // ----------------------------------------

    function downloadCSV() {
        if (drawingData.length === 0) {
            alert("データがありません。");
            return;
        }

        // CSVのヘッダー行
        const headers = "timestamp,event_type,stroke_id,x,y,pressure,tilt_x,tilt_y";
        
        // CSVのデータ行を作成
        const csvRows = drawingData.map(d => {
            return [
                d.timestamp,
                d.event_type,
                d.stroke_id,
                d.x.toFixed(2), // 小数点以下2桁に丸める
                d.y.toFixed(2),
                d.pressure.toFixed(4),
                d.tilt_x,
                d.tilt_y
            ].join(',');
        });

        // ヘッダーとデータ行を結合
        const csvContent = headers + "\n" + csvRows.join("\n");

        // UTF-8 BOM を追加（Excelでの文字化け対策）
        const bom = new Uint8Array([0xEF, 0xBB, 0xBF]);
        const blob = new Blob([bom, csvContent], { type: 'text/csv;charset=utf-8;' });
        
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = 'unpitsu_data.csv'; // ダウンロードファイル名
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
    }

    function clearCanvas() {
        if (confirm("データを消去しますか？")) {
            drawingData = [];
            strokeCount = 0;
            ctx.clearRect(0, 0, canvas.width, canvas.height);
        }
    }
});