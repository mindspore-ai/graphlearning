import { describe, beforeEach, test, it, expect, beforeAll } from "vitest";
import {
    getImageDimensionsFromBase64Async,
    getAreaBySymbolOuterRadius,
} from "../src/utils/otherUtils";
import * as otherUtils from "../src/utils/otherUtils";
import { Window, HTMLElement, GlobalWindow } from "happy-dom";
import jsdom from "jsdom";

describe("test getImageDimensionsFromBase64Async", () => {
    const LOAD_FAILURE_SRC = "LOAD_FAILURE_SRC";
    const LOAD_SUCCESS_SRC = "LOAD_SUCCESS_SRC";
    const mockBase64 =
        "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAEgAAABICAMAAABiM0N1AAAC+lBMVEUAAABqnP9kj/96af96af9Cov96af96av8/p/9Ziv86qv97af9qfP9JoP9pf/9Gqf8+pv83rf9Im/83rP96aP94bP9Xjv84rP9sef9nf/83r/9/c/83rf9wdP9Im/96aP9gg/9QlP9xcv9ApP96af97af9biP83rf9Gnv9Apv94bf9InP9Uj/9kf/96af9sdf9Qlf91bv9DpP9Skv9Wkv9cif93cP86sf9pef92bf9oe/93bf97aP9jgv83rP89p/85rP9Vjv9nfP84rf////8pMuHc3P96aP/P6f/a3v/Z4P/R5//V5P/N7f/O7P/P6//W4f/T5v/V4//U5f/M7//Y4f86qP/6+f/R6v/X4/85qv/b3//j4/89pv9Cof93a/9Ao//6/f84q/9Jmv9sd//R6f/T6P9fhP9jf//9//88p/9TkP8/pP9Ll/9Qkv9Vjv9udP9ycP82rP9Fnf9HnP9OlP9Ziv9hgv9lff9pef9Xi/9chv9wcv/8/f/i8f/l5f9En/9Nlv9ne/90bv91bf/X8f/1+v9biP+GkvLb8v/x8P9wee729f/w9f/Q4//z+f/r+P/q6v+3wPqorvePofRJU+fr9f/n6P+WzP/G1P2tu/qUmvSCi/FhautOWOg7ROUtNuIpMuLd3f/c1//N0/6jsPebpvaXoPV5g+9TXelBS+Y1PuQxOuP1/P/g9f/m8v/q7//f7f/g6P/a5f/Z4f/Q3f/Tzf+zqv9/bv/NzP7Hxf6+0Py5xvuJjvJvfe5nc+1eZ+ssNeLl9//s7P/n6//L5f/V3//b2/+02f+cw/97w//IwP9fuP9UsP9Grf98mv9kmf95if+GeP/I2v62uvyMlfN+ifFXYurQ8f/n5f/X5f/V1v+iu/9wu/93uv9nuf+Etf+vs/+Erf+Wpv+uov9Lof9xnf+Jmf+YmP+dkP9oif9xfP+81f7D0PyuxPmerPa94P+f1f+Sr/8/q/+Cpf9epP+goP+ilf9XlP+Zi/+Hi/+0y/t8gO9seO0TTIgyAAAARHRSTlMABw3x/LOxi1/r5F1CLyMY7+7u1cxycGpqWT4W+vDi4NbU0dC8oJCOinkx+fLv6rqwlW5eRjs6J/nn5uHczMK8ubl5Ts8VVsgAAAfnSURBVFjDnZh1cJNBEMWDu7u7u7snkLRIIBQS3N1dW6yFQmmhxd3d3d3d3d3dZYZ9e3fJ91ESCu+vZDr5ze673b3bGjwpXsyc0bKkT5wnT+L0mQrmLBLP8D9KUDhawiZNmgzp2bNXr5YtO3ce2rt3l7wFkyf4N0qkmBU6dWrWjDhDFAegLh07dq1UJFKEMZFzxW7cmEBOTkvFIVDXAbGSR44YJ0XCDh0aN0Y8SIw5nTWcAX36xEoeAUzqJA3qEYcTe3hzypFg30khIZPWBr84eu+L5IwcOTBjsb+Zk6NBPck5OaVNc6NOzevcOMOggQPbts3u0aoSSRqIeDrdMhv/pPYb7wlO277l4ntIKxFxADrma3Srde8A6tu3Xb647jgx07Zgzm27M4CQcasC/f39AwNXjwtob5Synm7bl0Dt+qX6M6cKcSixDlMGS0v2+deoUaNmzVo1a5lrmc1me+Ak+ZfB04hDoGEp/5hX2h5I7NQbI2tlYN0a4BCoVi2z3W42tzZ7t/GV/i94Rpx+w/r/IaYSiXogsRO+IppVdevWVRwzRCBv7zbebVYL1LqzxBnWv2Q4nxLEZs7NALZmX10XhwMijreZON61a9dezWaNfUwB9W+V7rezi5SkRwsCnWBOc3/FIQl7zK1bMwayreWg6j8GqFVmfT3l6IGATo5jd64wh8ThAIO8FMhms60P4JjOEqdVt0I6o8FpcIr9GccUlz/gAEQChlS9evWxRlKj58TpNqi4BpQEnPd2xXH5IzHeKh5gSHXqMGlBq24Eyqzpd07sLefFGJ3PJA5HJkYYkoOzu0CcQd2TOedPIgR0YjB8viLDIX9cAcEgUDggBlkd6+H44G/E6Z5OzadcfGL+OHd/FY8yiOuQJcKRHKvDsQ5VUJU4w4erkGKjNY4h1H2/cUQ9K39kPBunOqwOq4/PGvziE4FGlIokehUGdfBFYlqOWfoDjIYTenmvafZiq4/VugHJ+RFnRNM4DCrvDAh9ocIB5tUE1sJgaTNpxwoTFObj4+V1AL95MnxE06ZZuTlgUL2DODHps7Jnl0lqabDz3GeZWDO9oPr0o+sjCBQ1Dc4enJOAu/od9tinmZxaaIOIM3WuRINjmYzzOde0abVqsLsihtkU4ZBu/lx2gXZxPACZhPYA1LAhKmYzQNEJlIhAje2oaf38eWVyaZmqH5XaGHAsltH0s/kARaH7HQGdAtpfP39mmjR6qQpox0T+Pt2rIXEsfpgCW6qRchtS8JRGsvq++D5XC1os4qE6PBS6aNGiMItQQxTlU4DiGHLg1oBFIfr5s9uk1W4CAUP1Q+fO/gBUNQgmARTDEA2gI7BIN3+272XAcnl0s7lRrVYqaIUBpypMOk4ccjsL7sODqEbX/HHV0AzbMvFhB/qLOF4LLy2ZMQ0UBo2H2wAlNfDDwxdVpJ0/L+aYoBXnbT8F6CISo764hC9zFzOGBLevApTBkBAPj0k4NO38WWJiLaltWyQLx8EBbVjO32aBAzWiH+4HKL+BHwwBAGnmz3nRUXOm2myylpcfsvqQPwsEdiI4ChQEUFQDv8xCANLMn0uyoFGGE8XnaXxeoTI+PQgyJAQIqQW65s9BkcDe7QDJLGfwcYXJwgZEeTQanCiG2HhRrQXINX9GyeLhvlgo29QK0GJJVSC0rR9AZQ1Z8FT0pu++zmsneLaJz2YH98XGOcKvjQDNEKDpkqM9/mgAoSAnme1yPk+QM0e211LxNQyFOEZ+ViAU5DZRkDnx5ESLBNjl/RUsazCUOQ6rjGKJl6WhRXbyBQWq72qRmHi63jKS7PI6lT5cRkBW0jR55GgMOUW2Ss48NO1X0bTxhhDoYXOYJO8Led7LxkCvqd9XCMsWUBXOFsd5WHNog3mM0KMkMZaE2ngUCM5rk1ZzqJz3iI+LCCSMn621CF5jsBmi4XF/FOjWfH8hepeWEuiiLB6LZav4NEtlNlhZhFFbGEvCRyNyw/W13aRTGNVzqMzVYpGfZjqriHSOLcJ1xEtLMKZ/m3CgiTx/yBnoetVt4sMPdWYqM7qOkBtAdzgkXIPa0bgi1AuaLqDUX3z8y7cpq9WgzSqubN5++MrGPRi8cJTS9JdWMQ8vXNy5c/oC+u3hMPqgDj9IDiNkBkVKj+2HQ1pNJDnm5fwBR41DJW0RPRBTTSgX1rEu50FaK+8vpujnczhNdt5FydRDKzHWsTMBSO6a8/3D88dDQFxEm7iInItgcl7rjiKkkGvEQVoIx308KrPPAGUzOJWJ19X1/Oh1IB68f8ABxA2okbM9NK//ogw6s4ZjcpA8BuQ6fD/ZHi5l53X1ET96A/j+YrnDQOOVRUl1K0SmrthXHwXwcrDOR+TliQOQumS1ih+L994P9Y3Q6A2a+92N9itQIYNO8WLx/nx6jVizDnjOy5UaDk2vYnl5f3421ciqv8ZDPKoe58lG06soBYS990ZzudCO9WvoAYRTGy9u69+VSi7ip1FQghU0ev9kv0Z/VBC/RMVr7XfFjUUc7L33DxgjpgfiBgmn+BnBwd573xIhkOq1cIqUHRzee8/exevXo/CAcBMSpZdRcLBmPn9yd9vh8aOD6rvVVgZFxRIZPqiU+QQI6yG2KCwbTat5VFI3/4iqXJoDwnoIzghwPCuNu//+pCogObxF/RUU1cP/gNKkLJAugomhSzwrd8oY0TOXiRL1L5gMMSigX5gXyTqSTkXWAAAAAElFTkSuQmCC5XaGHAsltH0s/kARaH7HQGdAtpfP39mmjR6qQpox0T+Pt2rIXEsfpgCW6qRchtS8JRGsvq++D5XC1os4qE6PBS6aNGiMItQQxTlU4DiGHLg1oBFIfr5s9uk1W4CAUP1Q+fO/gBUNQgmARTDEA2gI7BIN3+272XAcnl0s7lRrVYqaIUBpypMOk4ccjsL7sODqEbX/HHV0AzbMvFhB/qLOF4LLy2ZMQ0UBo2H2wAlNfDDwxdVpJ0/L+aYoBXnbT8F6CISo764hC9zFzOGBLevApTBkBAPj0k4NO38WWJiLaltWyQLx8EBbVjO32aBAzWiH+4HKL+BHwwBAGnmz3nRUXOm2myylpcfsvqQPwsEdiI4ChQEUFQDv8xCANLMn0uyoFGGE8XnaXxeoTI+PQgyJAQIqQW65s9BkcDe7QDJLGfwcYXJwgZEeTQanCiG2HhRrQXINX9GyeLhvlgo29QK0GJJVSC0rR9AZQ1Z8FT0pu++zmsneLaJz2YH98XGOcKvjQDNEKDpkqM9/mgAoSAnme1yPk+QM0e211LxNQyFOEZ+ViAU5DZRkDnx5ESLBNjl/RUsazCUOQ6rjGKJl6WhRXbyBQWq72qRmHi63jKS7PI6lT5cRkBW0jR55GgMOUW2Ss48NO1X0bTxhhDoYXOYJO8Led7LxkCvqd9XCMsWUBXOFsd5WHNog3mM0KMkMZaE2ngUCM5rk1ZzqJz3iI+LCCSMn621CF5jsBmi4XF/FOjWfH8hepeWEuiiLB6LZav4NEtlNlhZhFFbGEvCRyNyw/W13aRTGNVzqMzVYpGfZjqriHSOLcJ1xEtLMKZ/m3CgiTx/yBnoetVt4sMPdWYqM7qOkBtAdzgkXIPa0bgi1AuaLqDUX3z8y7cpq9WgzSqubN5++MrGPRi8cJTS9JdWMQ8vXNy5c/oC+u3hMPqgDj9IDiNkBkVKj+2HQ1pNJDnm5fwBR41DJW0RPRBTTSgX1rEu50FaK+8vpujnczhNdt5FydRDKzHWsTMBSO6a8/3D88dDQFxEm7iInItgcl7rjiKkkGvEQVoIx308KrPPAGUzOJWJ19X1/Oh1IB68f8ABxA2okbM9NK//ogw6s4ZjcpA8BuQ6fD/ZHi5l53X1ET96A/j+YrnDQOOVRUl1K0SmrthXHwXwcrDOR+TliQOQumS1ih+L994P9Y3Q6A2a+92N9itQIYNO8WLx/nx6jVizDnjOy5UaDk2vYnl5f3421ciqv8ZDPKoe58lG06soBYS990ZzudCO9WvoAYRTGy9u69+VSi7ip1FQghU0ev9kv0Z/VBC/RMVr7XfFjUUc7L33DxgjpgfiBgmn+BnBwd573xIhkOq1cIqUHRzee8/exevXo/CAcBMSpZdRcLBmPn9yd9vh8aOD6rvVVgZFxRIZPqiU+QQI6yG2KCwbTat5VFI3/4iqXJoDwnoIzghwPCuNu//+pCogObxF/RUU1cP/gNKkLJAugomhSzwrd8oY0TOXiRL1L5gMMSigX5gXyTqSTkXWAAAAAElFTkSuQmCC";

    it("should get width and height", async () => {
        const window = new Window({
            innerWidth: 1080,
            innerHeight: 768,
            url: "http://localhost:6000",
        });

        global.window = window;
        global.document = window.document;
        window.document.write(
            `<html>
            <head>
                 <title>Test page</title>
            </head>
            <body>
                <div>

                </div>
                </body>
                </html>`
        );

        console.log(Object.keys(global.Image));
        Object.defineProperty(global.Image.prototype, "src", {
            set(src) {
                if (src === LOAD_FAILURE_SRC) {
                    setTimeout(() => this.onerror(new Error("mocked error")));
                } else if (src === LOAD_SUCCESS_SRC || src === mockBase64) {
                    setTimeout(() => this.onload());
                }
            },
        });
        console.log(global.Image.prototype);
        const { width, height } = await getImageDimensionsFromBase64Async(
            mockBase64
        );

        console.log(width, height);

        // expect(width).toBe(72);
        // expect(height).toBe(72);
    });

    it("should reject with an error for invalid base64 data", async () => {
        const invalidBase64Data = ""; // 无效的 base64 数据

        try {
            await getImageDimensionsFromBase64Async(invalidBase64Data);
            // 如果成功执行到这里，表示测试失败
            throw new Error("Expected the function to reject.");
        } catch (error) {
            // 预期函数会拒绝，并且错误消息应该包含特定文本
            expect(error.message).toContain("failed to load img -><-");
        }
    });
});
describe("test path drawing funcs", () => {
    it("should calc correct circlePath", () => {
        const ret = otherUtils.circlePath(1);
        expect(ret).toEqual("M0 1 A1 1 0 1 1 0 -1 a1 1 0 0 1 0 2 z");
    });
    it("should calc correct leftHalfCirclePath", () => {
        const ret = otherUtils.leftHalfCirclePath(1);
        expect(ret).toEqual("M0 1 A1 1 0 1 1 0 -1 v2 z");
    });
    it("should calc correct rightHalfCirclePath", () => {
        const ret = otherUtils.rightHalfCirclePath(1);
        expect(ret).toEqual("M0 1 A1 1 0 1 0 0 -1 v2 z");
    });
    it("should calc correct rightPercentCirclePath", () => {
        const ret = otherUtils.rightPercentCirclePath(1, 0.4);
        expect(ret).toEqual(
            "M0 0 L0.30901699437494745 0.9510565162951535 A1 1 0 0 0 0.30901699437494745 -0.9510565162951535 L0 0 z"
        );
    });
    it("should calc correct leftPercentCirclePath", () => {
        const ret = otherUtils.leftPercentCirclePath(1, 0.6);
        expect(ret).toEqual(
            "M0 0 L0.30901699437494734 0.9510565162951536 A1 1 0 1 1 0.30901699437494734 -0.9510565162951536 L0 0 z"
        );
    });
    it("should calc correct trianglePath", () => {
        const ret = otherUtils.trianglePath(1);
        expect(ret).toEqual(
            "M 0 -1 L -0.8660254037844386 0.5 L 0.8660254037844386 0.5 L 0 -1 z"
        );
    });
    it("should calc correct leftHalfTrianglePath", () => {
        const ret = otherUtils.leftHalfTrianglePath(1);
        expect(ret).toEqual("M0 -1 V0.5 L-0.8660254037844386 0.5 L0 -1 z");
    });
    it("should calc correct rightHalfTrianglePath", () => {
        const ret = otherUtils.rightHalfTrianglePath(1);
        expect(ret).toEqual("M0 -1 V0.5 L0.8660254037844386 0.5 L0 -1 z");
    });
    it("should calc correct rectPath", () => {
        const ret = otherUtils.rectPath(1);
        expect(ret).toEqual(
            "M-0.7071067811865475 -0.7071067811865475 L 0.7071067811865475 -0.7071067811865475 L 0.7071067811865475 0.7071067811865475 L -0.7071067811865475 0.7071067811865475 L -0.7071067811865475 -0.7071067811865475 z"
        );
    });
    it("should calc correct leftHalfRectPath", () => {
        const ret = otherUtils.leftHalfRectPath(1);
        expect(ret).toEqual(
            "M0 -0.7071067811865475 L 0 0.7071067811865475 L -0.7071067811865475 0.7071067811865475 L -0.7071067811865475 -0.7071067811865475 L 0 -0.7071067811865475 z"
        );
    });
    it("should calc correct rightHalfRectPath", () => {
        const ret = otherUtils.rightHalfRectPath(1);
        expect(ret).toEqual(
            "M0 -0.7071067811865475 L 0 0.7071067811865475 L 0.7071067811865475 0.7071067811865475 L 0.7071067811865475 -0.7071067811865475 L 0 -0.7071067811865475 z"
        );
    });
    it("should calc correct crossPath", () => {
        const ret = otherUtils.crossPath(1);
        expect(ret).toEqual(
            "M0.3333333333333333 -1 v0.6666666666666666 h0.6666666666666666 v0.6666666666666666 h-0.6666666666666666 v0.6666666666666666 h-0.6666666666666666 v-0.6666666666666666 h-0.6666666666666666 v-0.6666666666666666 h0.6666666666666666 v-0.6666666666666666 h0.6666666666666666 z"
        );
    });
    it("should calc correct leftHalfCrossPath", () => {
        const ret = otherUtils.leftHalfCrossPath(1);
        expect(ret).toEqual(
            "M0 -1 V1 H-0.3333333333333333 V0.3333333333333333 H-1 V-0.3333333333333333 H-0.3333333333333333 V-1 H0 z"
        );
    });
    it("should calc correct rightHalfCrossPath", () => {
        const ret = otherUtils.rightHalfCrossPath(1);
        expect(ret).toEqual(
            "M0 -1 V1 H0.3333333333333333 V0.3333333333333333 H1 V-0.3333333333333333 H0.3333333333333333 V-1 H0 z"
        );
    });
    it("should calc correct diamondPath", () => {
        const ret = otherUtils.diamondPath(1);
        expect(ret).toEqual(
            "M0 -1 L-0.5773502691896258 0 L0 1 L0.5773502691896258 0 L0 -1 z"
        );
    });
    it("should calc correct leftHalfDiamondPath", () => {
        const ret = otherUtils.leftHalfDiamondPath(1);
        expect(ret).toEqual("M0 -1 L-0.5773502691896258 0 L0 1 L0 -1 z");
    });
    it("should calc correct rightHalfDiamondPath", () => {
        const ret = otherUtils.rightHalfDiamondPath(1);
        expect(ret).toEqual("M0 -1 L0.5773502691896258 0 L0 1 L0 -1 z");
    });
    it("should calc correct wyePath", () => {
        const ret = otherUtils.wyePath(1);
        expect(ret).toEqual(
            "M0 -0.4480184754795918 L0.6720277132193876 -0.8360138566096939 l0.38799538113010207 0.6720277132193876  L0.38799538113010207 0.2240092377397959 v0.7759907622602041 h-0.7759907622602041 v-0.7759907622602041 l-0.6720277132193876 -0.38799538113010207 L-0.6720277132193876 -0.8360138566096939 L0 -0.4480184754795918 z"
        );
    });
    it("should calc correct leftHalfWyePath", () => {
        const ret = otherUtils.leftHalfWyePath(1);
        expect(ret).toEqual(
            "M0 -0.4480184754795918 L-0.6720277132193876 -0.8360138566096939 l-0.38799538113010207 0.6720277132193876  L-0.38799538113010207 0.2240092377397959 v0.7759907622602041 h0.38799538113010207 V-0.4480184754795918z"
        );
    });
    it("should calc correct rightHalfWyePath", () => {
        const ret = otherUtils.rightHalfWyePath(1);
        expect(ret).toEqual(
            "M0 -0.4480184754795918 L0.6720277132193876 -0.8360138566096939 l0.38799538113010207 0.6720277132193876  L0.38799538113010207 0.2240092377397959 v0.7759907622602041 h-0.38799538113010207 V-0.4480184754795918z"
        );
    });
    it("should calc correct starPath", () => {
        const ret = otherUtils.starPath(1);
        expect(ret).toEqual(
            "M0 -1 L0.22451398828979266 -0.3090169943749474 H0.9510565162951535 L0.3632712640026804 0.11803398874989482 L0.5877852522924731 0.8090169943749475 L0 0.3819660112501051 L-0.5877852522924731 0.8090169943749475 L-0.3632712640026804 0.11803398874989482 L-0.9510565162951535 -0.3136359283204443 L-0.22451398828979266 -0.3090169943749474 z"
        );
    });
    it("should calc correct leftHalfStarPath", () => {
        const ret = otherUtils.leftHalfStarPath(1);
        expect(ret).toEqual(
            "M0 -1 L-0.22451398828979266 -0.3090169943749474 H-0.9510565162951535 L-0.3632712640026804 0.11803398874989482 L-0.5877852522924731 0.8090169943749475 L0 0.3819660112501051 L0 -1 z"
        );
    });
    it("should calc correct rightHalfStarPath", () => {
        const ret = otherUtils.rightHalfStarPath(1);
        expect(ret).toEqual(
            "M0 -1 L0.22451398828979266 -0.3090169943749474 H0.9510565162951535 L0.3632712640026804 0.11803398874989482 L0.5877852522924731 0.8090169943749475 L0 0.3819660112501051 L0 -1 z"
        );
    });
    it("should calc correct eksPath", () => {
        const ret = otherUtils.eksPath(1);
        expect(ret).toEqual(
            "M0 -0.4714045207910316 L0.4714045207910316 -0.9428090415820634 l0.4714045207910316 0.4714045207910316 L0.4714045207910316 0 l0.4714045207910316 0.4714045207910316 L0.4714045207910316 0.9428090415820634 L0 0.4714045207910316 L-0.4714045207910316 0.9428090415820634 l-0.4714045207910316 -0.4714045207910316 L-0.4714045207910316 0 l-0.4714045207910316 -0.4714045207910316 L-0.4714045207910316 -0.9428090415820634 L0 -0.4714045207910316 z"
        );
    });
    it("should calc correct leftHalfEksPath", () => {
        const ret = otherUtils.leftHalfEksPath(1);
        expect(ret).toEqual(
            "M0 -0.4714045207910316 L-0.4714045207910316 -0.9428090415820634 l-0.4714045207910316 0.4714045207910316 L-0.4714045207910316 0 l-0.4714045207910316 0.4714045207910316 L-0.4714045207910316 0.9428090415820634 L0 0.4714045207910316 L0 -0.4714045207910316 z"
        );
    });
    it("should calc correct rightHalfEksPath", () => {
        const ret = otherUtils.rightHalfEksPath(1);
        expect(ret).toEqual(
            "M0 -0.4714045207910316 L0.4714045207910316 -0.9428090415820634 l0.4714045207910316 0.4714045207910316 L0.4714045207910316 0 l0.4714045207910316 0.4714045207910316 L0.4714045207910316 0.9428090415820634 L0 0.4714045207910316 L0 -0.4714045207910316 z"
        );
    });
    it("should calc correct leftHalfAndWholeCirclePathStroke", () => {
        const ret = otherUtils.leftHalfAndWholeCirclePathStroke(1);
        expect(ret).toEqual(
            "M0 1 A1 1 0 1 0 0 -1 v2 z A1 1 0 1 1 0 -1 a1 1 0 0 1 0 2 z"
        );
    });
    it("should calc correct rightHalfAndWholeCirclePathStroke", () => {
        const ret = otherUtils.rightHalfAndWholeCirclePathStroke(1);
        expect(ret).toEqual(
            "M0 1 A1 1 0 1 1 0 -1 v2 z A1 1 0 1 0 0 -1 a1 1 0 0 0 0 2 z"
        );
    });
    it("should calc correct leftHalfAndWholeCirclePathFill", () => {
        const ret = otherUtils.leftHalfAndWholeCirclePathFill(1);
        expect(ret).toEqual(
            "M0 0 A0 0 0 1 0 0 0 v1 z m0 1 A1 1 0 1 1 0 -1 a1 1 0 0 1 0 2 z"
        );
    });
    it("should calc correct rightHalfAndWholeCirclePathFill", () => {
        const ret = otherUtils.rightHalfAndWholeCirclePathFill(1);
        expect(ret).toEqual(
            "M0 0 A0 0 0 1 1 0 0 v1 z m0 1 A1 1 0 1 0 0 -1 a1 1 0 0 0 0 2 z"
        );
    });
});

describe("test getAreaBySymbolOuterRadius", () => {
    it("should calc correctly on circle", () => {
        const ret = getAreaBySymbolOuterRadius("circle", 1);
        expect(ret).toBeCloseTo(3.14159, 4);
    });
    it("should calc correctly on cross", () => {
        const ret = getAreaBySymbolOuterRadius("cross", 1);
        expect(ret).toBeCloseTo(2.22222, 4);
    });
    it("should calc correctly on diamond", () => {
        const ret = getAreaBySymbolOuterRadius("diamond", 1);
        expect(ret).toBeCloseTo(1.1547, 4);
    });
    it("should calc correctly on square", () => {
        const ret = getAreaBySymbolOuterRadius("square", 1);
        expect(ret).toEqual(2);
    });
    it("should calc correctly on triangle", () => {
        const ret = getAreaBySymbolOuterRadius("triangle", 1);
        expect(ret).toBeCloseTo(1.299038, 4);
    });
    it("should calc correctly on star", () => {
        const ret = getAreaBySymbolOuterRadius("star", 1);
        expect(ret).toBeCloseTo(0.890813, 4);
    });
    it("should calc correctly on wye", () => {
        const ret = getAreaBySymbolOuterRadius("wye", 1);
        expect(ret).toBeCloseTo(2.06722, 4);
    });
    it("should return 0 on unknown type", () => {
        const ret = getAreaBySymbolOuterRadius("type", 1);
        expect(ret).toEqual(0);
    });
});
