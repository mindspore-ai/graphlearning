import { createTestingPinia } from "@pinia/testing";
import ElementPlus from "element-plus";
import "element-plus/dist/index.css";
import { describe, it, expect, vi } from "vitest";
import { mount } from "@vue/test-utils";

import TreeRenderer from "../../../src/components/footer/tree.vue";
import { Dataset, SingleDashboard, View } from "../../../src/types/types";
import { useMyStore } from "../../../src/stores/store";
import { getImageDimensionsFromBase64Async } from "../../../src/utils/otherUtils.ts";

const props: InstanceType<typeof TreeRenderer>["$props"] = {};

const testPinia = createTestingPinia({
    createSpy: vi.fn,
    stubActions: true,
    initialState: {
        my: {
            // ...myStore,

            datasetList: [
                {
                    name: "dataset-test",
                    taskType: "node-classification",
                    predLabels: [1, 2, 3, 0, 3, 2, 1],
                    trueLabels: [0, 2, 1, 3, 0, 2, 2],
                    numNodeClasses: 4,
                } as Partial<Dataset>,
            ],
            singleDashboardList: [
                {
                    id: "db-test-0",
                    refDatasetName: "dataset-test",
                    isRoot: true,
                    labelType: "true",
                    isHighlightCorrespondingNode: true,
                    parentId: "",
                } as Partial<SingleDashboard>,
                {
                    id: "db-test-1",
                    refDatasetName: "dataset-test",
                    isRoot: false,
                    labelType: "true",
                    isHighlightCorrespondingNode: true,
                    parentId: "db-test-0",
                } as Partial<SingleDashboard>,
                {
                    id: "db-test-2",
                    refDatasetName: "dataset-test",
                    isRoot: false,
                    labelType: "true",
                    isHighlightCorrespondingNode: true,
                    parentId: "db-test-0",
                } as Partial<SingleDashboard>,
                {
                    id: "db-test-3",
                    refDatasetName: "dataset-test",
                    isRoot: true,
                    labelType: "true",
                    isHighlightCorrespondingNode: true,
                    parentId: "db-test-1",
                } as Partial<SingleDashboard>,
            ],
            getPrincipalViewOfDashboard: vi.fn(),
            // .mockImplementation(
            // (db: Partial<SingleDashboard>) =>
            //     ({
            //         snapshotBase64:
            //             "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAEgAAABICAMAAABiM0N1AAAC+lBMVEUAAABqnP9kj/96af96af9Cov96af96av8/p/9Ziv86qv97af9qfP9JoP9pf/9Gqf8+pv83rf9Im/83rP96aP94bP9Xjv84rP9sef9nf/83r/9/c/83rf9wdP9Im/96aP9gg/9QlP9xcv9ApP96af97af9biP83rf9Gnv9Apv94bf9InP9Uj/9kf/96af9sdf9Qlf91bv9DpP9Skv9Wkv9cif93cP86sf9pef92bf9oe/93bf97aP9jgv83rP89p/85rP9Vjv9nfP84rf////8pMuHc3P96aP/P6f/a3v/Z4P/R5//V5P/N7f/O7P/P6//W4f/T5v/V4//U5f/M7//Y4f86qP/6+f/R6v/X4/85qv/b3//j4/89pv9Cof93a/9Ao//6/f84q/9Jmv9sd//R6f/T6P9fhP9jf//9//88p/9TkP8/pP9Ll/9Qkv9Vjv9udP9ycP82rP9Fnf9HnP9OlP9Ziv9hgv9lff9pef9Xi/9chv9wcv/8/f/i8f/l5f9En/9Nlv9ne/90bv91bf/X8f/1+v9biP+GkvLb8v/x8P9wee729f/w9f/Q4//z+f/r+P/q6v+3wPqorvePofRJU+fr9f/n6P+WzP/G1P2tu/qUmvSCi/FhautOWOg7ROUtNuIpMuLd3f/c1//N0/6jsPebpvaXoPV5g+9TXelBS+Y1PuQxOuP1/P/g9f/m8v/q7//f7f/g6P/a5f/Z4f/Q3f/Tzf+zqv9/bv/NzP7Hxf6+0Py5xvuJjvJvfe5nc+1eZ+ssNeLl9//s7P/n6//L5f/V3//b2/+02f+cw/97w//IwP9fuP9UsP9Grf98mv9kmf95if+GeP/I2v62uvyMlfN+ifFXYurQ8f/n5f/X5f/V1v+iu/9wu/93uv9nuf+Etf+vs/+Erf+Wpv+uov9Lof9xnf+Jmf+YmP+dkP9oif9xfP+81f7D0PyuxPmerPa94P+f1f+Sr/8/q/+Cpf9epP+goP+ilf9XlP+Zi/+Hi/+0y/t8gO9seO0TTIgyAAAARHRSTlMABw3x/LOxi1/r5F1CLyMY7+7u1cxycGpqWT4W+vDi4NbU0dC8oJCOinkx+fLv6rqwlW5eRjs6J/nn5uHczMK8ubl5Ts8VVsgAAAfnSURBVFjDnZh1cJNBEMWDu7u7u7snkLRIIBQS3N1dW6yFQmmhxd3d3d3d3d3dZYZ9e3fJ91ESCu+vZDr5ze673b3bGjwpXsyc0bKkT5wnT+L0mQrmLBLP8D9KUDhawiZNmgzp2bNXr5YtO3ce2rt3l7wFkyf4N0qkmBU6dWrWjDhDFAegLh07dq1UJFKEMZFzxW7cmEBOTkvFIVDXAbGSR44YJ0XCDh0aN0Y8SIw5nTWcAX36xEoeAUzqJA3qEYcTe3hzypFg30khIZPWBr84eu+L5IwcOTBjsb+Zk6NBPck5OaVNc6NOzevcOMOggQPbts3u0aoSSRqIeDrdMhv/pPYb7wlO277l4ntIKxFxADrma3Srde8A6tu3Xb647jgx07Zgzm27M4CQcasC/f39AwNXjwtob5Synm7bl0Dt+qX6M6cKcSixDlMGS0v2+deoUaNmzVo1a5lrmc1me+Ak+ZfB04hDoGEp/5hX2h5I7NQbI2tlYN0a4BCoVi2z3W42tzZ7t/GV/i94Rpx+w/r/IaYSiXogsRO+IppVdevWVRwzRCBv7zbebVYL1LqzxBnWv2Q4nxLEZs7NALZmX10XhwMijreZON61a9dezWaNfUwB9W+V7rezi5SkRwsCnWBOc3/FIQl7zK1bMwayreWg6j8GqFVmfT3l6IGATo5jd64wh8ThAIO8FMhms60P4JjOEqdVt0I6o8FpcIr9GccUlz/gAEQChlS9evWxRlKj58TpNqi4BpQEnPd2xXH5IzHeKh5gSHXqMGlBq24Eyqzpd07sLefFGJ3PJA5HJkYYkoOzu0CcQd2TOedPIgR0YjB8viLDIX9cAcEgUDggBlkd6+H44G/E6Z5OzadcfGL+OHd/FY8yiOuQJcKRHKvDsQ5VUJU4w4erkGKjNY4h1H2/cUQ9K39kPBunOqwOq4/PGvziE4FGlIokehUGdfBFYlqOWfoDjIYTenmvafZiq4/VugHJ+RFnRNM4DCrvDAh9ocIB5tUE1sJgaTNpxwoTFObj4+V1AL95MnxE06ZZuTlgUL2DODHps7Jnl0lqabDz3GeZWDO9oPr0o+sjCBQ1Dc4enJOAu/od9tinmZxaaIOIM3WuRINjmYzzOde0abVqsLsihtkU4ZBu/lx2gXZxPACZhPYA1LAhKmYzQNEJlIhAje2oaf38eWVyaZmqH5XaGHAsltH0s/kARaH7HQGdAtpfP39mmjR6qQpox0T+Pt2rIXEsfpgCW6qRchtS8JRGsvq++D5XC1os4qE6PBS6aNGiMItQQxTlU4DiGHLg1oBFIfr5s9uk1W4CAUP1Q+fO/gBUNQgmARTDEA2gI7BIN3+272XAcnl0s7lRrVYqaIUBpypMOk4ccjsL7sODqEbX/HHV0AzbMvFhB/qLOF4LLy2ZMQ0UBo2H2wAlNfDDwxdVpJ0/L+aYoBXnbT8F6CISo764hC9zFzOGBLevApTBkBAPj0k4NO38WWJiLaltWyQLx8EBbVjO32aBAzWiH+4HKL+BHwwBAGnmz3nRUXOm2myylpcfsvqQPwsEdiI4ChQEUFQDv8xCANLMn0uyoFGGE8XnaXxeoTI+PQgyJAQIqQW65s9BkcDe7QDJLGfwcYXJwgZEeTQanCiG2HhRrQXINX9GyeLhvlgo29QK0GJJVSC0rR9AZQ1Z8FT0pu++zmsneLaJz2YH98XGOcKvjQDNEKDpkqM9/mgAoSAnme1yPk+QM0e211LxNQyFOEZ+ViAU5DZRkDnx5ESLBNjl/RUsazCUOQ6rjGKJl6WhRXbyBQWq72qRmHi63jKS7PI6lT5cRkBW0jR55GgMOUW2Ss48NO1X0bTxhhDoYXOYJO8Led7LxkCvqd9XCMsWUBXOFsd5WHNog3mM0KMkMZaE2ngUCM5rk1ZzqJz3iI+LCCSMn621CF5jsBmi4XF/FOjWfH8hepeWEuiiLB6LZav4NEtlNlhZhFFbGEvCRyNyw/W13aRTGNVzqMzVYpGfZjqriHSOLcJ1xEtLMKZ/m3CgiTx/yBnoetVt4sMPdWYqM7qOkBtAdzgkXIPa0bgi1AuaLqDUX3z8y7cpq9WgzSqubN5++MrGPRi8cJTS9JdWMQ8vXNy5c/oC+u3hMPqgDj9IDiNkBkVKj+2HQ1pNJDnm5fwBR41DJW0RPRBTTSgX1rEu50FaK+8vpujnczhNdt5FydRDKzHWsTMBSO6a8/3D88dDQFxEm7iInItgcl7rjiKkkGvEQVoIx308KrPPAGUzOJWJ19X1/Oh1IB68f8ABxA2okbM9NK//ogw6s4ZjcpA8BuQ6fD/ZHi5l53X1ET96A/j+YrnDQOOVRUl1K0SmrthXHwXwcrDOR+TliQOQumS1ih+L994P9Y3Q6A2a+92N9itQIYNO8WLx/nx6jVizDnjOy5UaDk2vYnl5f3421ciqv8ZDPKoe58lG06soBYS990ZzudCO9WvoAYRTGy9u69+VSi7ip1FQghU0ev9kv0Z/VBC/RMVr7XfFjUUc7L33DxgjpgfiBgmn+BnBwd573xIhkOq1cIqUHRzee8/exevXo/CAcBMSpZdRcLBmPn9yd9vh8aOD6rvVVgZFxRIZPqiU+QQI6yG2KCwbTat5VFI3/4iqXJoDwnoIzghwPCuNu//+pCogObxF/RUU1cP/gNKkLJAugomhSzwrd8oY0TOXiRL1L5gMMSigX5gXyTqSTkXWAAAAAElFTkSuQmCC5XaGHAsltH0s/kARaH7HQGdAtpfP39mmjR6qQpox0T+Pt2rIXEsfpgCW6qRchtS8JRGsvq++D5XC1os4qE6PBS6aNGiMItQQxTlU4DiGHLg1oBFIfr5s9uk1W4CAUP1Q+fO/gBUNQgmARTDEA2gI7BIN3+272XAcnl0s7lRrVYqaIUBpypMOk4ccjsL7sODqEbX/HHV0AzbMvFhB/qLOF4LLy2ZMQ0UBo2H2wAlNfDDwxdVpJ0/L+aYoBXnbT8F6CISo764hC9zFzOGBLevApTBkBAPj0k4NO38WWJiLaltWyQLx8EBbVjO32aBAzWiH+4HKL+BHwwBAGnmz3nRUXOm2myylpcfsvqQPwsEdiI4ChQEUFQDv8xCANLMn0uyoFGGE8XnaXxeoTI+PQgyJAQIqQW65s9BkcDe7QDJLGfwcYXJwgZEeTQanCiG2HhRrQXINX9GyeLhvlgo29QK0GJJVSC0rR9AZQ1Z8FT0pu++zmsneLaJz2YH98XGOcKvjQDNEKDpkqM9/mgAoSAnme1yPk+QM0e211LxNQyFOEZ+ViAU5DZRkDnx5ESLBNjl/RUsazCUOQ6rjGKJl6WhRXbyBQWq72qRmHi63jKS7PI6lT5cRkBW0jR55GgMOUW2Ss48NO1X0bTxhhDoYXOYJO8Led7LxkCvqd9XCMsWUBXOFsd5WHNog3mM0KMkMZaE2ngUCM5rk1ZzqJz3iI+LCCSMn621CF5jsBmi4XF/FOjWfH8hepeWEuiiLB6LZav4NEtlNlhZhFFbGEvCRyNyw/W13aRTGNVzqMzVYpGfZjqriHSOLcJ1xEtLMKZ/m3CgiTx/yBnoetVt4sMPdWYqM7qOkBtAdzgkXIPa0bgi1AuaLqDUX3z8y7cpq9WgzSqubN5++MrGPRi8cJTS9JdWMQ8vXNy5c/oC+u3hMPqgDj9IDiNkBkVKj+2HQ1pNJDnm5fwBR41DJW0RPRBTTSgX1rEu50FaK+8vpujnczhNdt5FydRDKzHWsTMBSO6a8/3D88dDQFxEm7iInItgcl7rjiKkkGvEQVoIx308KrPPAGUzOJWJ19X1/Oh1IB68f8ABxA2okbM9NK//ogw6s4ZjcpA8BuQ6fD/ZHi5l53X1ET96A/j+YrnDQOOVRUl1K0SmrthXHwXwcrDOR+TliQOQumS1ih+L994P9Y3Q6A2a+92N9itQIYNO8WLx/nx6jVizDnjOy5UaDk2vYnl5f3421ciqv8ZDPKoe58lG06soBYS990ZzudCO9WvoAYRTGy9u69+VSi7ip1FQghU0ev9kv0Z/VBC/RMVr7XfFjUUc7L33DxgjpgfiBgmn+BnBwd573xIhkOq1cIqUHRzee8/exevXo/CAcBMSpZdRcLBmPn9yd9vh8aOD6rvVVgZFxRIZPqiU+QQI6yG2KCwbTat5VFI3/4iqXJoDwnoIzghwPCuNu//+pCogObxF/RUU1cP/gNKkLJAugomhSzwrd8oY0TOXiRL1L5gMMSigX5gXyTqSTkXWAAAAAElFTkSuQmCC",
            //     } as any)
            // )//NOTE 在这mock的不管用
        },
    },
});
vi.mock("../../../src/utils/otherUtils.ts", async (importOriginal) => {
    const mod = await importOriginal();
    return {
        ...mod,
        getImageDimensionsFromBase64Async: () =>
            new Promise<{
                width: number;
                height: number;
            }>((resolve, reject) => {
                resolve({ width: 50, height: 50 });
            }),
    };
});

describe("test history tree renderer", () => {
    it("test render", async () => {
        const myStore = useMyStore();
        // console.log(myStore.getPrincipalViewOfDashboard(13));//undefined

        myStore.getPrincipalViewOfDashboard.mockImplementation(() => ({
            snapshotBase64:
                "data:image/png;base64,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",
        }));

        // console.log(myStore.getPrincipalViewOfDashboard(13));// { snapshotBase64: "data:image/png;...

        const wrapper = mount(TreeRenderer, {
            props: props,
            global: {
                plugins: [testPinia, ElementPlus],
            },
        });
        console.log(wrapper.html());
        expect(wrapper.html()).toContain("calculating tree");

        await new Promise<void>((resolve, reject) => {
            setTimeout(() => {
                resolve();
            }, 1000);
        });

        const nodes = wrapper.findAll(".tree-nodes rect");
        expect(nodes.length).toBe(4);
        const links = wrapper.findAll(".tree-links path");
        expect(links.length).toBe(3);

        wrapper.unmount();
    });
});
