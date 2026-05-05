// swift-tools-version: 5.9

import PackageDescription

let package = Package(
    name: "YunshuSDK",
    platforms: [.macOS(.v13), .iOS(.v16)],
    products: [
        .library(name: "YunshuSDK", targets: ["YunshuSDK"]),
    ],
    targets: [
        .target(name: "YunshuSDK"),
        .testTarget(name: "YunshuSDKTests", dependencies: ["YunshuSDK"]),
    ]
)
