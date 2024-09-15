// script.js

/**
 * @typedef {Object} IsaacItem
 * @property {string} name - The name of the item
 * @property {string} img_url - The URL to the item's image
 * @property {string} wiki_url - The URL to the item's wiki page
 * @property {string} description - The description of the item
 * @property {number} confidence - The confidence score of the detection
 * @property {number} [timestamp] - The timestamp when the item was detected
 */

/**
 * Socket.IO client instance
 * @type {SocketIOClient.Socket}
 */
const socket = io.connect();

/**
 * Cache to store recently detected items
 * @type {IsaacItem[]}
 */
let itemCache = [];

/**
 * Duration (in milliseconds) to keep items in the cache
 * @constant {number}
 */
const ITEM_CACHE_DURATION = 10000; // 10 seconds

/**
 * Updates the item cache with new items and removes old items
 * Prevents duplicates by checking for existing items
 *
 * @param {IsaacItem[]} newItems - Array of new items to add to the cache
 */
function updateItemCache(newItems) {
    const currentTime = Date.now();

    newItems.forEach(newItem => {
        // see if item already exists in the cache
        const existingItemIndex = itemCache.findIndex(item => item.name === newItem.name);

        if (existingItemIndex !== -1) {
            // item exists, update its timestamp
            itemCache[existingItemIndex].timestamp = currentTime;
            itemCache[existingItemIndex].confidence = newItem.confidence; // update the confidence if needed
        } else {
            // item does not exist, add it to the cache
            newItem.timestamp = currentTime;
            itemCache.push(newItem);
        }
    });

    // Remove items older than ITEM_CACHE_DURATION
    itemCache = itemCache.filter(item => (currentTime - item.timestamp) <= ITEM_CACHE_DURATION);

    // Update the item list display
    updateItemListDisplay();
}

/**
 * Updates the display of the item list on the frontend
 */
function updateItemListDisplay() {
    const itemList = document.getElementById('item-list');
    itemList.innerHTML = ''; // clear previous items

    itemCache.forEach(item => {
        const itemElement = document.createElement('div');
        itemElement.classList.add('item');

        // close button
        const closeButton = document.createElement('button');
        closeButton.classList.add('close-button');
        closeButton.innerHTML = '&times;'; // HTML entity for '×'
        closeButton.title = 'Remove item';
        closeButton.addEventListener('click', () => {
            removeItemFromCache(item.name);
        });
        itemElement.appendChild(closeButton);

        // item image
        const imgElement = document.createElement('img');
        imgElement.src = item.img_url;
        imgElement.alt = item.name;
        itemElement.appendChild(imgElement);

        // item name with hyperlink
        const nameElement = document.createElement('a');
        nameElement.href = item.wiki_url;
        nameElement.target = '_blank';
        nameElement.textContent = item.name;
        itemElement.appendChild(nameElement);

        // item description
        const descElement = document.createElement('p');
        descElement.textContent = item.description;
        itemElement.appendChild(descElement);

        // item confidence
        const confidenceElement = document.createElement('p');
        confidenceElement.textContent = `Confidence: ${item.confidence.toFixed(2)}`;
        confidenceElement.classList.add('confidence');
        itemElement.appendChild(confidenceElement);

        itemList.appendChild(itemElement);
    });
}

/**
 * Removes an item from the cache and updates the display
 *
 * @param {string} itemName - The name of the item to remove
 */
function removeItemFromCache(itemName) {
    // Remove the item from the cache
    itemCache = itemCache.filter(item => item.name !== itemName);
    // Update the display
    updateItemListDisplay();
}

/**
 * Periodically cleans the item cache to remove old items
 */
setInterval(() => {
    const currentTime = Date.now();
    itemCache = itemCache.filter(item => (currentTime - item.timestamp) <= ITEM_CACHE_DURATION);
    updateItemListDisplay();
}, 1000);

/**
 * Client-side settings
 * These are double-defined unfortunately between python and js for now
 * @type {Object}
 */
const clientSettings = {
    confidenceThreshold: 0.6,
    bboxColor: '#00FF00',
    bboxLabelColor: '#000000'
};

/**
 * Sends the updated settings to the server
 */
function sendSettingsToServer() {
    socket.emit('update_settings', clientSettings);
}

/**
 * Initializes the controls and sets up event listeners
 */
function initControls() {
    const confidenceInput = document.getElementById('confidence-threshold');
    const bboxColorInput = document.getElementById('bbox-color');
    const bboxLabelColorInput = document.getElementById('bbox-label-color');

    confidenceInput.addEventListener('input', () => {
        let value = parseFloat(confidenceInput.value);
        if (isNaN(value) || value < 0 || value > 1.0) {
            value = 0.6;
        }
        clientSettings.confidenceThreshold = value;
        console.log(`Calling sendSettingsToServer for confidenceThreshold=${clientSettings.confidenceThreshold}`);
        sendSettingsToServer();
    });

    bboxColorInput.addEventListener('input', () => {
        clientSettings.bboxColor = bboxColorInput.value;
        console.log(`Calling sendSettingsToServer for bboxColor=${clientSettings.bboxColor}`);
        sendSettingsToServer();
    });

    bboxLabelColorInput.addEventListener('input', () => {
        clientSettings.bboxLabelColor = bboxLabelColorInput.value;
        console.log(`Calling sendSettingsToServer for bboxLabelColor=${clientSettings.bboxLabelColor}`);
        sendSettingsToServer();
    });

    // send initial settings to server
    sendSettingsToServer();
}

// call initControls after the DOM content is loaded
document.addEventListener('DOMContentLoaded', initControls);

/**
 * Handles the 'inference_update' event from the server
 *
 * @param {Object} data - The data received from the server
 * @param {string[]} data.images - Array of base64-encoded images
 * @param {IsaacItem[]} data.item_metadata - Array of detected item metadata
 */
socket.on('inference_update', function (data) {
    const imagesContainer = document.getElementById('images-container');
    imagesContainer.innerHTML = '';  // Clear previous images

    data.images.forEach((imgBase64) => {
        const imgElement = document.createElement('img');
        imgElement.src = 'data:image/png;base64,' + imgBase64;
        imagesContainer.appendChild(imgElement);
    });

    if (data.item_metadata && data.item_metadata.length > 0) {
        updateItemCache(data.item_metadata);
    }
});
